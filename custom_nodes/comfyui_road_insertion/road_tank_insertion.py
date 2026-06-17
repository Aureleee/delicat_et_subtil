"""
road_tank_insertion.py
======================
Nœuds pour l'insertion du tank dans la scène route.

  1. RoadWidthBBoxEstimator : point (x,y) + masque → 2 bboxes CENTRÉES sur le point
  2. TankRGBABBox           : tank RGBA → tight bbox autour des pixels non-transparents
  3. TankCompositor         : base + tank RGBA + bboxes → composite + masques

Le point (point_x, point_y) est TOUJOURS le centre des bboxes.
"""

import numpy as np
import cv2
import torch


# ─── helpers ──────────────────────────────────────────────────────────────────

def _t2np(t):
    if t.dim() == 4: t = t[0]
    return (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)

def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

def _green_mask(rgb, thr=80):
    r = rgb[:,:,0].astype(np.int16)
    g = rgb[:,:,1].astype(np.int16)
    b = rgb[:,:,2].astype(np.int16)
    return ((g > r + thr) & (g > b + thr) & (g > 80)).astype(np.uint8) * 255

def _centered_bbox(cx, cy, w, h, W, H):
    """Bbox centrée sur (cx,cy), clampée à l'image."""
    x1 = max(0,   cx - w // 2)
    y1 = max(0,   cy - h // 2)
    x2 = min(W-1, cx + w // 2)
    y2 = min(H-1, cy + h // 2)
    return (x1, y1, x2, y2)

def _expand_bbox_centered(cx, cy, w, h, k, W, H):
    """Bbox élargie par facteur k, toujours centrée sur (cx,cy)."""
    return _centered_bbox(cx, cy, int(w * k), int(h * k), W, H)

def _draw_bbox(img, bbox, color, thick=2, label=""):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (x1+4, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (x1+4, y1+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

def _composite_rgba_on_rgb(base_bgr, rgba_src, dst_x1, dst_y1):
    H, W = base_bgr.shape[:2]
    sh, sw = rgba_src.shape[:2]
    sx1, sy1, sx2, sy2 = 0, 0, sw, sh
    dx1, dy1 = dst_x1, dst_y1
    dx2, dy2 = dst_x1 + sw, dst_y1 + sh
    if dx1 < 0: sx1 -= dx1; dx1 = 0
    if dy1 < 0: sy1 -= dy1; dy1 = 0
    if dx2 > W: sx2 -= (dx2 - W); dx2 = W
    if dy2 > H: sy2 -= (dy2 - H); dy2 = H
    if dx1 >= dx2 or dy1 >= dy2: return base_bgr
    crop    = rgba_src[sy1:sy2, sx1:sx2]
    alpha   = crop[:,:,3:4].astype(np.float32) / 255.0
    fg_bgr  = crop[:,:,2::-1].astype(np.float32)
    bg_crop = base_bgr[dy1:dy2, dx1:dx2].astype(np.float32)
    blended = (alpha * fg_bgr + (1 - alpha) * bg_crop).clip(0, 255).astype(np.uint8)
    base_bgr[dy1:dy2, dx1:dx2] = blended
    return base_bgr


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RoadWidthBBoxEstimator
# ═══════════════════════════════════════════════════════════════════════════════

class RoadWidthBBoxEstimator:
    """
    Calcule la largeur de la route au point (point_x, point_y) puis génère
    deux bboxes CENTRÉES sur ce point.

    Si une depth_map est fournie, la taille du tank est corrigée par la depth :
      scale = (d_ref / d_point) ^ depth_power
    où d_ref est la depth à la ligne de référence (proche caméra = bas de la route)
    et d_point est la depth au point du tank.
    Blanc = proche (d=1), Noir = loin (d=0).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image":  ("IMAGE",),
                "point_x":     ("INT", {"default": 0, "min": 0, "max": 8192}),
                "point_y":     ("INT", {"default": 0, "min": 0, "max": 8192}),
                "k_tank": ("FLOAT", {
                    "default": 1.2, "min": 0.1, "max": 5.0, "step": 0.05,
                    "tooltip": "Largeur tank = k_tank x largeur route (avant correction depth)."}),
                "height_ratio": ("FLOAT", {
                    "default": 0.75, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Ratio hauteur/largeur de la bbox tank."}),
                "k_inpaint": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Zone inpaint = k_inpaint x bbox_tank."}),
                "neighborhood_rows": ("INT", {
                    "default": 7, "min": 1, "max": 51, "step": 2,
                    "tooltip": "Lignes ±N autour du point pour moyenner la largeur route."}),
                "green_threshold": ("INT", {"default": 80, "min": 10, "max": 200, "step": 5}),
            },
            "optional": {
                "depth_map": ("IMAGE", {
                    "tooltip": "Carte de profondeur (blanc=proche, noir=loin). "
                               "Utilisée pour réduire la taille du tank proportionnellement "
                               "à la distance. Connecter ici la sortie de DepthAnything/DepthPro."}),
                "depth_ref_y_frac": ("FLOAT", {
                    "default": 0.85, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction Y (0=haut, 1=bas) du point de référence depth "
                               "(= là où le tank est à taille normale, proche caméra). "
                               "Défaut 0.85 = bas de l'image."}),
                "depth_power": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Exposant de la correction depth. "
                               "1.0 = correction linéaire, >1 = réduction plus agressive."}),
            }
        }

    RETURN_TYPES  = ("FLOAT", "TANK_BBOX", "TANK_BBOX", "IMAGE", "FLOAT")
    RETURN_NAMES  = ("road_width_px", "tank_bbox", "inpaint_bbox", "debug_image", "depth_scale")
    FUNCTION      = "estimate"
    CATEGORY      = "road/insertion"

    def estimate(self, mask_image, point_x, point_y,
                 k_tank, height_ratio, k_inpaint,
                 neighborhood_rows, green_threshold,
                 depth_map=None, depth_ref_y_frac=0.85, depth_power=1.0):

        rgb   = _t2np(mask_image)
        H, W  = rgb.shape[:2]
        gmask = _green_mask(rgb, green_threshold)
        px, py = int(point_x), int(point_y)

        # ── Largeur route : scan horizontal ±N lignes ─────────────────────────
        half_n = neighborhood_rows // 2
        left_edges, right_edges = [], []

        for row in range(py - half_n, py + half_n + 1):
            if not (0 <= row < H):
                continue
            col = min(px, W-1)
            if gmask[row, col] == 0:
                continue
            left = px
            while left > 0 and gmask[row, left - 1] > 0:
                left -= 1
            right = px
            while right < W - 1 and gmask[row, right + 1] > 0:
                right += 1
            left_edges.append(left)
            right_edges.append(right)

        if left_edges:
            mean_left  = float(np.mean(left_edges))
            mean_right = float(np.mean(right_edges))
            road_width = mean_right - mean_left
        else:
            road_width = 100.0
            mean_left  = px - 50.0
            mean_right = px + 50.0

        road_width = max(road_width, 10.0)

        # ── Correction depth ──────────────────────────────────────────────────
        depth_scale = 1.0
        depth_info  = ""
        if depth_map is not None:
            # Convertir depth map en float [0,1] (blanc=proche=1, noir=loin=0)
            d_img = _t2np(depth_map)
            if d_img.ndim == 3:
                d_gray = d_img.mean(axis=2)
            else:
                d_gray = d_img.astype(np.float32)
            d_norm = d_gray.astype(np.float32) / 255.0

            # Resize si nécessaire
            if d_norm.shape != (H, W):
                d_norm = cv2.resize(d_norm, (W, H), interpolation=cv2.INTER_LINEAR)

            # depth au point du tank
            py_c = int(np.clip(py, 0, H - 1))
            px_c = int(np.clip(px, 0, W - 1))
            d_point = float(d_norm[py_c, px_c])

            # depth de référence = échantillon sur une ligne proche caméra
            ref_y = int(np.clip(depth_ref_y_frac * H, 0, H - 1))
            # Moyenne sur une bande horizontale de ±5% de H autour de ref_y,
            # uniquement sur les pixels de route (mask > 0)
            band = max(1, int(0.03 * H))
            y0r = max(0, ref_y - band)
            y1r = min(H, ref_y + band)
            road_band = gmask[y0r:y1r, :]
            d_band    = d_norm[y0r:y1r, :]
            road_pts  = d_band[road_band > 0]
            if len(road_pts) > 5:
                d_ref = float(np.median(road_pts))
            else:
                d_ref = float(d_norm[ref_y, :].mean())

            d_ref   = max(d_ref,   0.01)
            d_point = max(d_point, 0.01)

            # scale = (d_point / d_ref)^power
            # Si d_point < d_ref → tank plus loin → plus petit
            depth_scale = float((d_point / d_ref) ** depth_power)
            depth_scale = float(np.clip(depth_scale, 0.05, 2.0))
            depth_info  = f" | depth: ref={d_ref:.2f} pt={d_point:.2f} scale={depth_scale:.2f}"

        # ── BBoxes avec correction depth ──────────────────────────────────────
        tank_w = int(k_tank * road_width * depth_scale)
        tank_h = int(tank_w * height_ratio)
        tank_w = max(tank_w, 4)
        tank_h = max(tank_h, 4)
        tank_bbox    = _centered_bbox(px, py, tank_w, tank_h, W, H)
        inpaint_bbox = _expand_bbox_centered(px, py, tank_w, tank_h, k_inpaint, W, H)

        # ── Debug image ────────────────────────────────────────────────────────
        # Si depth map dispo : afficher en fond la depth colorisée
        if depth_map is not None:
            d_vis = (d_norm * 255).astype(np.uint8)
            d_color = cv2.applyColorMap(d_vis, cv2.COLORMAP_INFERNO)
            # Blend avec le masque route
            debug = d_color.copy()
            debug[gmask > 0] = np.clip(
                debug[gmask > 0].astype(np.float32) * 0.5
                + np.array([20, 200, 60]) * 0.5, 0, 255).astype(np.uint8)
        else:
            debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            debug[gmask > 0] = np.clip(
                debug[gmask > 0].astype(np.float32) * 0.5
                + np.array([30, 200, 60]) * 0.5, 0, 255).astype(np.uint8)

        cv2.line(debug, (int(mean_left), py), (int(mean_right), py),
                 (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(debug, f"road: {road_width:.0f}px x{depth_scale:.2f} = {tank_w}px",
                    (int(mean_left), max(py - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        if depth_map is not None:
            ref_y = int(depth_ref_y_frac * H)
            cv2.line(debug, (0, ref_y), (W, ref_y), (180, 100, 255), 1, cv2.LINE_AA)
            cv2.putText(debug, "depth ref", (4, ref_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 100, 255), 1, cv2.LINE_AA)

        _draw_bbox(debug, inpaint_bbox, (80,  80, 255), 2, "inpaint")
        _draw_bbox(debug, tank_bbox,    (0,  180, 255), 2, "tank")
        cv2.circle(debug, (px, py), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(debug, (px, py), 6, (0,   0,   0),   1, cv2.LINE_AA)

        cv2.putText(debug, f"scale={depth_scale:.2f}{depth_info}"[:80],
                    (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 200), 1, cv2.LINE_AA)

        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return (float(road_width), tank_bbox, inpaint_bbox, _np2t(debug_rgb), float(depth_scale))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TankRGBABBox
# ═══════════════════════════════════════════════════════════════════════════════

class TankRGBABBox:
    """
    Extrait la tight bbox des pixels non-transparents du rendu Blender.
    Permet de connaître les vraies dimensions du tank dans l'image RGBA.

    Outputs :
      tight_bbox      : (x1,y1,x2,y2) autour des pixels visibles
      content_width   : largeur du contenu en pixels
      content_height  : hauteur du contenu en pixels
      aspect_ratio    : width/height (> 1 = tank plus large que haut)
      debug_image     : RGBA avec tight_bbox tracée
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tank_rgba":       ("IMAGE",),
                "alpha_threshold": ("INT", {
                    "default": 10, "min": 1, "max": 254, "step": 1,
                    "tooltip": "Seuil alpha : pixels avec alpha > seuil = visibles."}),
            }
        }

    RETURN_TYPES  = ("TANK_BBOX", "INT", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES  = ("tight_bbox", "content_width", "content_height", "aspect_ratio", "debug_image")
    FUNCTION      = "extract"
    CATEGORY      = "road/insertion"

    def extract(self, tank_rgba, alpha_threshold):
        arr = _t2np(tank_rgba)   # HxWx3 ou HxWx4
        H, W = arr.shape[:2]

        if arr.shape[2] == 4:
            alpha = arr[:,:,3]
        else:
            # Pas de canal alpha : on prend tout
            alpha = np.ones((H, W), np.uint8) * 255

        mask = alpha > alpha_threshold

        if not np.any(mask):
            # Image vide : fallback bbox pleine
            tight_bbox = (0, 0, W-1, H-1)
            cw, ch = W, H
        else:
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
            x1, x2 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
            tight_bbox = (x1, y1, x2, y2)
            cw = x2 - x1
            ch = y2 - y1

        aspect = float(cw) / float(ch) if ch > 0 else 1.0

        # ── Debug : RGBA avec bbox tracée ──────────────────────────────────────
        if arr.shape[2] == 4:
            debug = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            # Fond quadrillé gris pour voir la transparence
            checker = np.zeros((H, W, 3), np.uint8)
            sz = 16
            for y in range(0, H, sz):
                for x in range(0, W, sz):
                    c = 180 if ((x // sz + y // sz) % 2 == 0) else 120
                    checker[y:y+sz, x:x+sz] = c
            a = arr[:,:,3:4].astype(np.float32) / 255.0
            fg  = arr[:,:,:3][:,:,::-1].astype(np.float32)
            debug = (a * fg + (1-a) * checker.astype(np.float32)).clip(0,255).astype(np.uint8)
        else:
            debug = arr[:,:,::-1].copy()

        _draw_bbox(debug, tight_bbox, (0, 200, 255), 2, f"{cw}x{ch}")
        cv2.putText(debug, f"ratio {aspect:.2f}",
                    (tight_bbox[0]+4, tight_bbox[3]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)

        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        return (tight_bbox, int(cw), int(ch), aspect, _np2t(debug_rgb))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TankCompositor
# ═══════════════════════════════════════════════════════════════════════════════

class TankCompositor:
    """
    Insère le tank RGBA dans la scène, centré sur (point_x, point_y),
    redimensionné pour que son COTE LARGE occupe la largeur de tank_bbox.

    Outputs :
      composited_image : base + tank composité
      inpaint_mask     : masque blanc dans inpaint_bbox
      tank_rgba_scaled : canvas taille inpaint_bbox avec tank à la bonne échelle
      debug_image      : composite + bboxes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image":   ("IMAGE",),
                "tank_rgba":    ("IMAGE",),
                "point_x":      ("INT", {"default": 0}),
                "point_y":      ("INT", {"default": 0}),
                "tank_bbox":    ("TANK_BBOX",),
                "inpaint_bbox": ("TANK_BBOX",),
            },
            "optional": {
                "y_offset_px": ("INT", {
                    "default": 0, "min": -500, "max": 500, "step": 1,
                    "tooltip": "Décalage vertical du tank en pixels. "
                               "Négatif = vers le haut, Positif = vers le bas."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("composited_image", "inpaint_mask", "tank_rgba_scaled", "debug_image")
    FUNCTION      = "composite"
    CATEGORY      = "road/insertion"

    def composite(self, base_image, tank_rgba,
                  point_x, point_y, tank_bbox, inpaint_bbox,
                  y_offset_px=0):

        base_rgb = _t2np(base_image)
        tank_arr = _t2np(tank_rgba)
        H, W = base_rgb.shape[:2]

        if tank_arr.shape[2] == 3:
            alpha_ch = np.ones((*tank_arr.shape[:2], 1), np.uint8) * 255
            tank_arr = np.concatenate([tank_arr, alpha_ch], axis=2)

        # ── Dimensions tank_bbox ───────────────────────────────────────────────
        tx1, ty1, tx2, ty2 = tank_bbox
        bbox_w = max(1, tx2 - tx1)
        bbox_h = max(1, ty2 - ty1)

        # ── Resize tank : le côté large du tank occupe le côté large de la bbox
        th, tw = tank_arr.shape[:2]
        # On scale pour que le tank tienne DANS la bbox (sans dépasser)
        scale  = min(bbox_w / max(tw, 1), bbox_h / max(th, 1))
        new_w  = max(1, int(tw * scale))
        new_h  = max(1, int(th * scale))
        tank_resized = cv2.resize(tank_arr, (new_w, new_h),
                                  interpolation=cv2.INTER_LANCZOS4)

        # ── Centrer sur (point_x, point_y) ────────────────────────────────────
        off_x = int(point_x) - new_w // 2
        off_y = int(point_y) - new_h // 2 + int(y_offset_px)

        # ── Composite sur base ─────────────────────────────────────────────────
        result_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR).copy()
        result_bgr = _composite_rgba_on_rgb(result_bgr, tank_resized, off_x, off_y)
        composited_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # ── Masque inpaint ─────────────────────────────────────────────────────
        ix1, iy1, ix2, iy2 = inpaint_bbox
        inpaint_mask = np.zeros((H, W, 3), np.uint8)
        inpaint_mask[iy1:iy2+1, ix1:ix2+1] = 255

        # ── Tank RGBA dans canvas taille inpaint_bbox ─────────────────────────
        canvas_w = max(1, ix2 - ix1)
        canvas_h = max(1, iy2 - iy1)
        tank_canvas = np.zeros((canvas_h, canvas_w, 4), np.uint8)

        rel_off_x = off_x - ix1
        rel_off_y = off_y - iy1
        sx1_, sy1_ = 0, 0
        sx2_, sy2_ = new_w, new_h
        dx1_, dy1_ = rel_off_x, rel_off_y
        dx2_, dy2_ = rel_off_x + new_w, rel_off_y + new_h
        if dx1_ < 0: sx1_ -= dx1_; dx1_ = 0
        if dy1_ < 0: sy1_ -= dy1_; dy1_ = 0
        if dx2_ > canvas_w: sx2_ -= (dx2_ - canvas_w); dx2_ = canvas_w
        if dy2_ > canvas_h: sy2_ -= (dy2_ - canvas_h); dy2_ = canvas_h
        if dx1_ < dx2_ and dy1_ < dy2_:
            tank_canvas[dy1_:dy2_, dx1_:dx2_] = \
                tank_resized[sy1_:sy2_, sx1_:sx2_]

        # ── Debug image ────────────────────────────────────────────────────────
        debug_bgr = result_bgr.copy()
        _draw_bbox(debug_bgr, inpaint_bbox, (80,  80, 255), 2, "inpaint")
        _draw_bbox(debug_bgr, tank_bbox,    (0,  180, 255), 2, "tank")
        debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)

        return (
            _np2t(composited_rgb),
            _np2t(inpaint_mask),
            _np2t(tank_canvas),
            _np2t(debug_rgb),
        )


# ─── Registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "RoadWidthBBoxEstimator": RoadWidthBBoxEstimator,
    "TankRGBABBox":           TankRGBABBox,
    "TankCompositor":         TankCompositor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RoadWidthBBoxEstimator": "Road Width BBox Estimator",
    "TankRGBABBox":           "Tank RGBA BBox",
    "TankCompositor":         "Tank Compositor",
}
