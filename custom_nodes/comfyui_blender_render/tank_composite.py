"""
TankComposite
=============
Superpose le tank RGBA sur le fond au point d'insertion, avec deux bboxes :
  - bbox_tank   : serrée autour du tank scalé (rectangle)
  - bbox_inpaint: carré centré sur le point, aire = K_inpaint × aire_tank_bbox

Outputs :
  debug_image  : fond + tank + les deux bboxes dessinées
  crop_image   : fond + tank cropé sur bbox_inpaint (propre, sans bbox)
  crop_x/y/size: coordonnées du crop dans l'image originale

InsertionGridDisplay
====================
Empile un batch de crops (IMAGE, B>1) en grille horizontale.
"""

import numpy as np
import torch
import cv2
from PIL import Image


ROAD_QUAD_PAIRS  = "ROAD_QUAD_PAIRS"
INSERTION_POINTS = "INSERTION_POINTS"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _t2np(t):
    """IMAGE tensor (1,H,W,C) → np.uint8 RGB ou RGBA."""
    if t.dim() == 4:
        t = t[0]
    return (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)

def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

def _alpha_bbox(rgba):
    """BBox serrée autour des pixels alpha > 10. Retourne (x, y, w, h) ou None."""
    if rgba.shape[2] < 4:
        return None
    alpha = rgba[:, :, 3]
    mask  = alpha > 10
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

def _mean_quad_width(quad_pairs):
    """Largeur moyenne géométrique de tous les quads (moyenne bord haut + bord bas)."""
    widths = []
    for e in (quad_pairs or []):
        q = np.asarray(e["quad"], dtype=np.float64)
        top_w = float(np.linalg.norm(q[1] - q[0]))
        bot_w = float(np.linalg.norm(q[2] - q[3]))
        w = (top_w + bot_w) / 2.0
        if w > 1.0:
            widths.append(w)
    return float(np.mean(widths)) if widths else 0.0


def _relative_scale_factor(road_px, mean_w, K_relative):
    """
    Deux sens : route étroite → tank plus grand, route large → tank plus petit.
    Formule : (mean_w / road_px) ^ K_relative
      road_px < mean_w  → facteur > 1  (tank agrandi)
      road_px > mean_w  → facteur < 1  (tank réduit)
      road_px == mean_w → facteur = 1  (neutre)
    K_relative=0 désactivé.
    """
    if K_relative == 0.0 or mean_w < 1.0 or road_px < 1.0:
        return 1.0
    return float((mean_w / road_px) ** K_relative)


def _road_width_at_point(quad_entry, perp):
    """
    Largeur de la route mesurée avec la perpendiculaire `perp` (même vecteur
    que celui utilisé pour mesurer tank_true_w → ratio cohérent).
    Projette les côtés gauche/droit du quad sur perp, médiane sur 9 positions.
    """
    q = np.asarray(quad_entry["quad"], dtype=np.float64)
    widths = []
    for t in np.linspace(0.1, 0.9, 9):
        lp = q[0] + t * (q[3] - q[0])
        rp = q[1] + t * (q[2] - q[1])
        w  = abs(float(np.dot(rp - lp, perp)))
        if w > 1.0:
            widths.append(w)
    return float(np.median(widths)) if widths else 0.0

def _scale_rgba(rgba_np, scale):
    """Redimensionne (H,W,4) par scale avec PIL LANCZOS."""
    H, W = rgba_np.shape[:2]
    nw, nh = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    pil = Image.fromarray(rgba_np, mode="RGBA")
    pil = pil.resize((nw, nh), Image.LANCZOS)
    return np.array(pil)

def _composite(bg_rgb, tank_rgba, cx, cy):
    """
    Colle tank_rgba sur bg_rgb. Le bas-centre du tank (alpha bbox) est aligné
    sur (cx, cy).  Retourne (composited_rgb, tank_x, tank_y, tank_w, tank_h).
    """
    out = bg_rgb.copy()
    bbox = _alpha_bbox(tank_rgba)
    if bbox is None:
        return out, 0, 0, 0, 0
    tx, ty, tw, th = bbox

    # Aligner centre de la bbox sur (cx, cy)
    ox = cx - (tx + tw // 2)
    oy = cy - (ty + th // 2)

    H, W = bg_rgb.shape[:2]
    th_full, tw_full = tank_rgba.shape[:2]

    # Zone de copie dans l'image de sortie
    dst_x0 = max(0, ox)
    dst_y0 = max(0, oy)
    dst_x1 = min(W, ox + tw_full)
    dst_y1 = min(H, oy + th_full)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return out, 0, 0, 0, 0

    src_x0 = dst_x0 - ox
    src_y0 = dst_y0 - oy
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    alpha_f = tank_rgba[src_y0:src_y1, src_x0:src_x1, 3:4].astype(np.float32) / 255.0
    rgb_t   = tank_rgba[src_y0:src_y1, src_x0:src_x1, :3].astype(np.float32)
    bg_r    = out[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = (rgb_t * alpha_f + bg_r * (1 - alpha_f)).clip(0, 255).astype(np.uint8)

    return out, ox + tx, oy + ty, tw, th   # coordonnées de la bbox serrée dans le fond


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 : TankComposite
# ═══════════════════════════════════════════════════════════════════════════════

class TankComposite:
    CATEGORY = "blender/composite"
    FUNCTION = "composite"
    RETURN_TYPES  = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("debug_image", "crop_image", "crop_x", "crop_y", "crop_size", "scale_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background":    ("IMAGE",),
                "tank_rgba":     ("IMAGE",),
                "active_quad":   (ROAD_QUAD_PAIRS,),
                "point_x":     ("INT",   {"default": 0, "min": 0, "max": 8192}),
                "point_y":     ("INT",   {"default": 0, "min": 0, "max": 8192}),
                "K_scale": ("FLOAT", {
                    "default": 0.65, "min": 0.01, "max": 5.0, "step": 0.01,
                    "tooltip": "scale = K × road_px / tank_true_width"}),
                "K_inpaint": ("FLOAT", {
                    "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "road_vector_2d": ("GRAVITY_FIELD", {
                    "tooltip": "Si branché : direction route depuis le champ de gravité. "
                               "Sinon : direction déduite de la géométrie du quad."}),
                "all_quad_pairs": (ROAD_QUAD_PAIRS, {
                    "tooltip": "Tous les quads de la scène — pour calculer la largeur moyenne "
                               "et pondérer le scale via K_relative."}),
                "K_relative": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Pondération par taille relative du quad.\n"
                               "0 = désactivé.\n"
                               "Route étroite (< moyenne) → tank agrandi.\n"
                               "Route large (> moyenne) → tank réduit.\n"
                               "Formule : scale × (mean_w / road_px)^K_relative"}),
                "show_bbox": ("BOOLEAN", {"default": False, "tooltip": "Afficher les bboxes et labels debug."}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500}),
                "tank_offset_x": ("INT", {"default": 0, "min": -500, "max": 500}),
                "tank_offset_y": ("INT", {"default": -5, "min": -500, "max": 500}),
                "K_tank_offset_y": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    def composite(self, background, tank_rgba, active_quad,
                  point_x, point_y,
                  K_scale=1.0, K_inpaint=4.0,
                  road_vector_2d=None, all_quad_pairs=None, K_relative=0.0,
                  show_bbox=True,
                  offset_x=0, offset_y=0,
                  tank_offset_x=0, tank_offset_y=0, K_tank_offset_y=0.0):

        bg   = _t2np(background)
        tank = _t2np(tank_rgba)
        H, W = bg.shape[:2]

        if tank.shape[2] == 3:
            alpha = np.full((*tank.shape[:2], 1), 255, np.uint8)
            tank  = np.concatenate([tank, alpha], axis=2)

        # ── 1. Direction de la route → perpendiculaire = direction largeur tank
        if road_vector_2d is not None:
            # Depuis le champ de gravité (sampé au point d'insertion)
            rv = road_vector_2d.cpu().float().numpy() if isinstance(road_vector_2d, torch.Tensor) \
                 else np.asarray(road_vector_2d, dtype=np.float32)
            if rv.ndim == 3:
                fy = int(np.clip(point_y, 0, rv.shape[1] - 1))
                fx = int(np.clip(point_x, 0, rv.shape[2] - 1))
                rv = rv[:, fy, fx]
            rv = rv[:2]
        else:
            # Fallback : depuis la géométrie du quad
            rv = _road_dir_from_quad(active_quad[0])
        n = float(np.linalg.norm(rv))
        rv = rv / n if n > 1e-6 else np.array([1.0, 0.0])
        perp = np.array([-rv[1], rv[0]])  # perpendiculaire = direction largeur

        # ── 2. Largeur RÉELLE du tank = span des pixels opaques sur perp ──────
        alpha_ch = tank[:, :, 3]
        ys, xs = np.where(alpha_ch > 10)
        if len(xs) > 0:
            proj = xs.astype(np.float32) * perp[0] + ys.astype(np.float32) * perp[1]
            tank_true_w = max(float(proj.max() - proj.min()), 1.0)
        else:
            tank_true_w = 1.0

        # ── 3. Scale : K × road_px / tank_true_w × facteur relatif ──────────
        road_px      = _road_width_at_point(active_quad[0], perp)
        mean_w       = _mean_quad_width(all_quad_pairs) if all_quad_pairs else 0.0
        rel_factor   = _relative_scale_factor(road_px, mean_w, K_relative)
        scale        = (K_scale * road_px) / tank_true_w * rel_factor
        tank_perp_px = tank_true_w  # for debug label

        tank_scaled = _scale_rgba(tank, scale)

        # ── 2. Offset proportionnel à la hauteur du tank scalé ────────────────
        scaled_bbox = _alpha_bbox(tank_scaled)
        scaled_h = scaled_bbox[3] if scaled_bbox else 1
        total_offset_y = tank_offset_y + int(round(K_tank_offset_y * scaled_h))

        # ── 3. Composite ──────────────────────────────────────────────────────
        composited, tbx, tby, tbw, tbh = _composite(bg, tank_scaled,
                                                      point_x + tank_offset_x,
                                                      point_y + total_offset_y)

        # ── 3. BBox inpaint (carré, centrée sur le point + offset) ───────────
        area_tank   = max(1, tbw * tbh)
        side_inp    = int(np.sqrt(K_inpaint * area_tank))
        cx_inp      = point_x + offset_x
        cy_inp      = point_y + offset_y
        ix0 = max(0, cx_inp - side_inp // 2)
        iy0 = max(0, cy_inp - side_inp // 2)
        ix1 = min(W, ix0 + side_inp)
        iy1 = min(H, iy0 + side_inp)
        # Recadrer si on dépasse
        ix0 = max(0, ix1 - side_inp)
        iy0 = max(0, iy1 - side_inp)
        actual_size = min(ix1 - ix0, iy1 - iy0)

        # ── 4. Debug ──────────────────────────────────────────────────────────
        debug = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
        scale_info = (f"road_px={road_px:.1f}  mean_w={mean_w:.1f}  tank_w={tank_perp_px:.1f}  "
                      f"K={K_scale}  rel={rel_factor:.3f}  scale={scale:.4f}")

        if show_bbox:
            q_pts = np.asarray(active_quad[0]["quad"], dtype=np.int32)
            cv2.polylines(debug, [q_pts], isClosed=True, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
            if tbw > 0:
                cv2.rectangle(debug, (tbx, tby), (tbx + tbw - 1, tby + tbh - 1),
                              (205, 90, 36), 2, cv2.LINE_AA)
                cv2.putText(debug, f"road={road_px:.0f}px  tank_w={tank_perp_px:.0f}px  K={K_scale}  scale={scale:.3f}",
                            (tbx, max(tby - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 90, 36), 1, cv2.LINE_AA)
            cv2.rectangle(debug, (ix0, iy0), (ix1 - 1, iy1 - 1), (0, 0, 220), 2, cv2.LINE_AA)
            cv2.putText(debug, f"inpaint {actual_size}x{actual_size}px",
                        (ix0 + 4, iy0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1, cv2.LINE_AA)
            cv2.circle(debug, (point_x, point_y), 3, (0, 0, 200), -1, cv2.LINE_AA)

        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)

        # ── 5. Crop (propre, sans bbox) ───────────────────────────────────────
        crop = composited[iy0:iy1, ix0:ix1]

        return (_np2t(debug_rgb), _np2t(crop), ix0, iy0, actual_size, scale_info)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 : InsertionPointSelector
# ═══════════════════════════════════════════════════════════════════════════════

class InsertionPointSelector:
    """
    Sélectionne un point d'insertion dans la liste INSERTION_POINTS par index.
    Sortie : point_x, point_y, active_quad (pour TankComposite ou RoadQuadGravitySampler).
    """
    CATEGORY = "blender/composite"
    FUNCTION = "select"
    RETURN_TYPES  = ("INT", "INT", ROAD_QUAD_PAIRS)
    RETURN_NAMES  = ("point_x", "point_y", "active_quad")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "insertion_points": (INSERTION_POINTS,),
                "index": ("INT", {"default": 1, "min": 0, "max": 64}),
            }
        }

    def select(self, insertion_points, index=0):
        if not insertion_points:
            return (0, 0, [])
        entry = insertion_points[min(index, len(insertion_points) - 1)]
        return (entry["x"], entry["y"], [entry["quad_entry"]])


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 : InsertionGridDisplay
# ═══════════════════════════════════════════════════════════════════════════════

class InsertionGridDisplay:
    """
    Prend un batch IMAGE (B, H, W, C) de crops et les affiche côte à côte.
    Si les crops ont des tailles différentes, ils sont redimensionnés à la
    hauteur du premier.
    """
    CATEGORY = "blender/composite"
    FUNCTION = "display"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("grid_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crops": ("IMAGE",),
            },
            "optional": {
                "padding": ("INT", {"default": 8, "min": 0, "max": 64}),
            }
        }

    def display(self, crops, padding=8):
        B = crops.shape[0]
        if B == 0:
            return (_np2t(np.zeros((64, 64, 3), np.uint8)),)

        imgs = []
        ref_h = None
        for i in range(B):
            img = (crops[i].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            if ref_h is None:
                ref_h = img.shape[0]
            if img.shape[0] != ref_h:
                scale = ref_h / img.shape[0]
                nw = max(1, int(img.shape[1] * scale))
                img = np.array(Image.fromarray(img).resize((nw, ref_h), Image.LANCZOS))
            imgs.append(img)

        pad = np.zeros((ref_h, padding, 3), np.uint8) if padding > 0 else None
        strips = []
        for i, img in enumerate(imgs):
            strips.append(img)
            if pad is not None and i < len(imgs) - 1:
                strips.append(pad)

        grid = np.concatenate(strips, axis=1)
        return (_np2t(grid),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 : AllTanksComposite
# ═══════════════════════════════════════════════════════════════════════════════

def _road_dir_from_quad(quad_entry):
    """Dérive la direction de la route depuis la géométrie du quad (normalisée)."""
    q = np.asarray(quad_entry["quad"], dtype=np.float64)
    # direction le long de la route = moyenne des deux côtés latéraux
    road_dir = ((q[3] - q[0]) + (q[2] - q[1])) / 2.0
    n = np.linalg.norm(road_dir)
    if n < 1e-6:
        return np.array([0.0, 1.0])
    return road_dir / n


class AllTanksComposite:
    """
    Composite tous les tanks sur le même fond.
    Pour chaque point d'insertion, recalcule depuis le quad associé :
      - l'orientation de la route
      - la perpendiculaire → vraie largeur du tank
      - la largeur de la route au point
      - le scale
    Fonctionne sur toutes les routes (tous les road_idx présents dans insertion_points).
    """
    CATEGORY = "blender/composite"
    FUNCTION = "composite_all"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("overview_image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background":       ("IMAGE",),
                "tank_rgba":        ("IMAGE",),
                "insertion_points": (INSERTION_POINTS,),
                "road_vector_2d":   ("GRAVITY_FIELD",),
                "K_scale":  ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0,  "step": 0.01}),
                "K_inpaint":("FLOAT", {"default": 4.0, "min": 1.0,  "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "all_quad_pairs": (ROAD_QUAD_PAIRS, {
                    "tooltip": "Tous les quads — pour K_relative (pondération par taille relative)."}),
                "K_relative": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "0=désactivé. Route étroite→tank plus grand, route large→tank plus petit."}),
                "tank_offset_x":   ("INT",   {"default": 0,   "min": -500, "max": 500}),
                "tank_offset_y":   ("INT",   {"default": 0,   "min": -500, "max": 500}),
                "K_tank_offset_y": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    def composite_all(self, background, tank_rgba, insertion_points, road_vector_2d,
                      K_scale=1.0, K_inpaint=4.0,
                      all_quad_pairs=None, K_relative=0.0,
                      tank_offset_x=0, tank_offset_y=0, K_tank_offset_y=0.0):

        canvas = _t2np(background).copy()
        H, W = canvas.shape[:2]

        tank = _t2np(tank_rgba)
        if tank.shape[2] == 3:
            alpha = np.full((*tank.shape[:2], 1), 255, np.uint8)
            tank  = np.concatenate([tank, alpha], axis=2)

        # road_vector_2d : champ (2, H, W) — direction locale de la route par pixel
        rv_field = road_vector_2d.cpu().float().numpy() if isinstance(road_vector_2d, torch.Tensor) \
                   else np.asarray(road_vector_2d, dtype=np.float32)

        alpha_ch = tank[:, :, 3]
        ys_all, xs_all = np.where(alpha_ch > 10)

        debug = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        for pt in (insertion_points or []):
            px, py = int(pt["x"]), int(pt["y"])
            quad_entry = pt["quad_entry"]

            # Direction locale de la route au point d'insertion (pixel exact)
            rv = rv_field.copy()
            if rv.ndim == 3:
                rv = rv[:, int(np.clip(py, 0, rv.shape[1]-1)),
                           int(np.clip(px, 0, rv.shape[2]-1))]
            rv = rv[:2]
            n = float(np.linalg.norm(rv))
            rv = rv / n if n > 1e-6 else np.array([1.0, 0.0])
            perp = np.array([-rv[1], rv[0]])

            # ── Vraie largeur du tank (projection sur perp) ───────────────────
            if len(xs_all) > 0:
                proj = xs_all.astype(np.float32) * perp[0] + ys_all.astype(np.float32) * perp[1]
                tank_true_w = max(float(proj.max() - proj.min()), 1.0)
            else:
                tank_true_w = 1.0

            road_px    = _road_width_at_point(quad_entry, perp)
            mean_w     = _mean_quad_width(all_quad_pairs) if all_quad_pairs else 0.0
            rel_factor = _relative_scale_factor(road_px, mean_w, K_relative)
            scale      = (K_scale * road_px) / tank_true_w * rel_factor

            tank_scaled = _scale_rgba(tank, scale)

            scaled_bbox = _alpha_bbox(tank_scaled)
            scaled_h = scaled_bbox[3] if scaled_bbox else 1
            total_offset_y = tank_offset_y + int(round(K_tank_offset_y * scaled_h))

            canvas_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
            composited, tbx, tby, tbw, tbh = _composite(canvas_rgb, tank_scaled,
                                                          px + tank_offset_x,
                                                          py + total_offset_y)
            debug = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)

            # Quad vert
            q_pts = np.asarray(quad_entry["quad"], dtype=np.int32)
            cv2.polylines(debug, [q_pts], isClosed=True, color=(0, 200, 0), thickness=1, lineType=cv2.LINE_AA)

            # bbox tank (bleu)
            if tbw > 0:
                cv2.rectangle(debug, (tbx, tby), (tbx + tbw - 1, tby + tbh - 1),
                              (205, 90, 36), 2, cv2.LINE_AA)

            # bbox inpaint (rouge)
            area_tank = max(1, tbw * tbh)
            side_inp  = int(np.sqrt(K_inpaint * area_tank))
            ix0 = int(np.clip(px - side_inp // 2, 0, W - 1))
            iy0 = int(np.clip(py - side_inp // 2, 0, H - 1))
            ix1 = int(np.clip(ix0 + side_inp, 0, W))
            iy1 = int(np.clip(iy0 + side_inp, 0, H))
            cv2.rectangle(debug, (ix0, iy0), (ix1 - 1, iy1 - 1), (0, 0, 220), 2, cv2.LINE_AA)

            # Point rouge
            cv2.circle(debug, (px, py), 3, (0, 0, 200), -1, cv2.LINE_AA)

            # Label
            label = f"r={road_px:.0f}px w={tank_true_w:.0f}px s={scale:.2f}"
            cv2.putText(debug, label, (tbx, max(tby - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (205, 90, 36), 1, cv2.LINE_AA)

        out_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return (_np2t(out_rgb),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 : BatchTankComposite
# ═══════════════════════════════════════════════════════════════════════════════

class BatchTankComposite:
    """
    Exactement TankComposite, mais boucle automatiquement sur tous les indices
    0 → len(insertion_points)-1. Produit un batch IMAGE (B, H, W, C) avec une
    image par position — à brancher directement sur SaveImage.
    """
    CATEGORY = "blender/composite"
    FUNCTION = "run"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images_batch",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background":       ("IMAGE",),
                "tank_rgba":        ("IMAGE",),
                "insertion_points": (INSERTION_POINTS,),
                "road_vector_2d":   ("GRAVITY_FIELD",),
                "K_scale":   ("FLOAT", {"default": 1.0, "min": 0.01, "max": 5.0,  "step": 0.01}),
                "K_inpaint": ("FLOAT", {"default": 4.0, "min": 1.0,  "max": 20.0, "step": 0.5}),
            },
            "optional": {
                "all_quad_pairs": (ROAD_QUAD_PAIRS, {
                    "tooltip": "Tous les quads — pour K_relative (pondération par taille relative)."}),
                "K_relative": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "0=désactivé. Route étroite→tank plus grand, route large→tank plus petit."}),
                "tank_offset_x":   ("INT",   {"default": 0,   "min": -500, "max": 500}),
                "tank_offset_y":   ("INT",   {"default": 0,   "min": -500, "max": 500}),
                "K_tank_offset_y": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    def run(self, background, tank_rgba, insertion_points, road_vector_2d,
            K_scale=1.0, K_inpaint=4.0,
            all_quad_pairs=None, K_relative=0.0,
            tank_offset_x=0, tank_offset_y=0, K_tank_offset_y=0.0):

        bg   = _t2np(background)
        H, W = bg.shape[:2]

        tank = _t2np(tank_rgba)
        if tank.shape[2] == 3:
            alpha = np.full((*tank.shape[:2], 1), 255, np.uint8)
            tank  = np.concatenate([tank, alpha], axis=2)

        rv_field = road_vector_2d.cpu().float().numpy() if isinstance(road_vector_2d, torch.Tensor) \
                   else np.asarray(road_vector_2d, dtype=np.float32)

        alpha_ch = tank[:, :, 3]
        ys_all, xs_all = np.where(alpha_ch > 10)

        frames = []

        for pt in (insertion_points or []):
            px, py     = int(pt["x"]), int(pt["y"])
            quad_entry = pt["quad_entry"]

            # Direction locale de la route au pixel exact du point d'insertion
            rv = rv_field.copy()
            if rv.ndim == 3:
                rv = rv[:, int(np.clip(py, 0, rv.shape[1]-1)),
                           int(np.clip(px, 0, rv.shape[2]-1))]
            rv = rv[:2]
            n = float(np.linalg.norm(rv))
            rv   = rv / n if n > 1e-6 else np.array([1.0, 0.0])
            perp = np.array([-rv[1], rv[0]])

            # Vraie largeur du tank
            if len(xs_all) > 0:
                proj = xs_all.astype(np.float32) * perp[0] + ys_all.astype(np.float32) * perp[1]
                tank_true_w = max(float(proj.max() - proj.min()), 1.0)
            else:
                tank_true_w = 1.0

            road_px    = _road_width_at_point(quad_entry, perp)
            mean_w     = _mean_quad_width(all_quad_pairs) if all_quad_pairs else 0.0
            rel_factor = _relative_scale_factor(road_px, mean_w, K_relative)
            scale      = (K_scale * road_px) / tank_true_w * rel_factor
            tank_scaled = _scale_rgba(tank, scale)

            scaled_bbox    = _alpha_bbox(tank_scaled)
            scaled_h       = scaled_bbox[3] if scaled_bbox else 1
            total_offset_y = tank_offset_y + int(round(K_tank_offset_y * scaled_h))

            composited, tbx, tby, tbw, tbh = _composite(bg, tank_scaled,
                                                          px + tank_offset_x,
                                                          py + total_offset_y)
            frames.append(composited)

        if not frames:
            frames.append(bg)

        batch = np.stack(frames, axis=0)                         # (B, H, W, 3)
        return (torch.from_numpy(batch.astype(np.float32) / 255.0),)


NODE_CLASS_MAPPINGS = {
    "TankComposite":           TankComposite,
    "InsertionPointSelector":  InsertionPointSelector,
    "InsertionGridDisplay":    InsertionGridDisplay,
    "AllTanksComposite":       AllTanksComposite,
    "BatchTankComposite":      BatchTankComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TankComposite":           "Tank Composite",
    "InsertionPointSelector":  "Insertion Point Selector",
    "InsertionGridDisplay":    "Insertion Grid Display",
    "AllTanksComposite":       "All Tanks Composite",
    "BatchTankComposite":      "Batch Tank Composite",
}
