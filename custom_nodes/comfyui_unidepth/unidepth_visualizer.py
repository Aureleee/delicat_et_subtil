"""
UniDepth Visualizer Node
========================
Prend l'output du node UniDepthInference et affiche les points sélectionnés
avec leur distance en mètres + incertitude (si confidence disponible).

Node 1 : UniDepthPointViz
  Affiche une grille de points sur l'image + leur distance caméra en mètres.
  Si la confidence_map est connectée, affiche aussi l'incertitude à chaque point.

Node 2 : UniDepthRoadWidth
  Prend le metric_depth + un masque de route + les LINE_SEGMENTS des bords
  (depuis RoadMaskBorderFit ou LocalBorderReconstruction) et estime la largeur
  réelle de la route en mètres à plusieurs hauteurs.
  Affiche les droites + distances estimées.
"""

import numpy as np
import torch
import cv2
from .unidepth_inference import UNIDEPTH_DEPTH, UNIDEPTH_POINTS, UNIDEPTH_INTRINSICS

# Type token partagé avec road_quad_segments
ROAD_QUAD_PAIRS = "ROAD_QUAD_PAIRS"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _t2np_img(t):
    if t.dim() == 4:
        t = t[0]
    return (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)

def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

def _get_depth_at(depth_np, x, y, radius=3):
    """Retourne (median_depth, std_depth) dans un patch radius×radius autour de (x,y)."""
    H, W = depth_np.shape
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    patch = depth_np[y0:y1, x0:x1]
    valid = patch[patch > 0.1]
    if valid.size == 0:
        return 0.0, 0.0
    return float(np.median(valid)), float(np.std(valid))

def _get_conf_at(conf_np, x, y, radius=3):
    if conf_np is None:
        return None
    H, W = conf_np.shape
    x0, x1 = max(0, x - radius), min(W, x + radius + 1)
    y0, y1 = max(0, y - radius), min(H, y + radius + 1)
    patch = conf_np[y0:y1, x0:x1]
    return float(patch.mean()) if patch.size > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 : Point Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

class UniDepthPointViz:
    """
    Affiche une grille (ou des points custom) sur l'image originale avec :
    - La distance en mètres à chaque point
    - La déviation standard locale (= proxy d'incertitude géométrique)
    - La confidence du modèle si elle est connectée (V2 only)

    Couleur des labels :
      Vert  → confidence élevée / faible std
      Orange → confidence moyenne
      Rouge  → faible confidence / forte incertitude (à prendre avec précaution)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "metric_depth": (UNIDEPTH_DEPTH,),
            },
            "optional": {
                "confidence_map": ("IMAGE", {
                    "tooltip": "Connecter la confidence_map du node UniDepthInference "
                               "(V2 uniquement). Colore les labels selon la confiance."}),
                "grid_cols": ("INT", {
                    "default": 6, "min": 1, "max": 20,
                    "tooltip": "Nombre de colonnes de la grille de points."}),
                "grid_rows": ("INT", {
                    "default": 4, "min": 1, "max": 20,
                    "tooltip": "Nombre de lignes de la grille de points."}),
                "margin_frac": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Marge (fraction de l'image) avant le premier point."}),
                "show_std": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Afficher la déviation standard locale (±Xm) = incertitude géométrique."}),
                "point_radius": ("INT", {"default": 3, "min": 1, "max": 20}),
                "font_scale": ("FLOAT", {"default": 0.45, "min": 0.2, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("viz_image",)
    FUNCTION      = "visualize"
    CATEGORY      = "depth/unidepth"

    def visualize(self, image, metric_depth,
                  confidence_map=None, grid_cols=6, grid_rows=4,
                  margin_frac=0.1, show_std=True, point_radius=3, font_scale=0.45):

        rgb   = _t2np_img(image)
        H, W  = rgb.shape[:2]

        # depth : (1,H,W) → (H,W)
        if metric_depth.dim() == 3:
            depth_np = metric_depth.squeeze(0).cpu().float().numpy()
        else:
            depth_np = metric_depth.cpu().float().numpy()

        # confidence : IMAGE (1,H,W,3) → (H,W) grayscale normalisée
        conf_np = None
        if confidence_map is not None:
            c = _t2np_img(confidence_map)
            conf_np = c.mean(axis=2).astype(np.float32) / 255.0

        # Resize depth si nécessaire
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)
        if conf_np is not None and conf_np.shape != (H, W):
            conf_np = cv2.resize(conf_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # ── Grille de points ──────────────────────────────────────────────────
        mx = int(margin_frac * W)
        my = int(margin_frac * H)
        xs = np.linspace(mx, W - mx, grid_cols).astype(int)
        ys = np.linspace(my, H - my, grid_rows).astype(int)

        vis = rgb.copy()

        for y in ys:
            for x in xs:
                d, std = _get_depth_at(depth_np, x, y, radius=point_radius + 2)
                conf = _get_conf_at(conf_np, x, y, radius=point_radius + 2)

                if d < 0.1:
                    continue

                # Couleur selon confidence / std
                if conf is not None:
                    if conf > 0.6:
                        color = (50, 220, 50)    # vert
                    elif conf > 0.3:
                        color = (255, 165, 0)    # orange
                    else:
                        color = (220, 50, 50)    # rouge
                else:
                    # Pas de confidence : couleur selon std relative
                    rel_std = std / max(d, 0.01)
                    if rel_std < 0.05:
                        color = (50, 220, 50)
                    elif rel_std < 0.15:
                        color = (255, 165, 0)
                    else:
                        color = (220, 50, 50)

                # Dessiner le point
                cv2.circle(vis, (x, y), point_radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(vis, (x, y), point_radius, color, -1, cv2.LINE_AA)

                # Label
                label = f"{d:.1f}m"
                if show_std and std > 0.01:
                    label += f" ±{std:.2f}"
                if conf is not None:
                    label += f" c={conf:.2f}"

                lx = x + point_radius + 3
                ly = y + 4

                # Fond noir pour lisibilité
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, 1)
                cv2.rectangle(vis, (lx - 1, ly - th - 2), (lx + tw + 1, ly + 2),
                              (0, 0, 0), -1)
                cv2.putText(vis, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)

        # Légende
        legend_y = H - 10
        if conf_np is not None:
            cv2.putText(vis, "vert=confiant  orange=moyen  rouge=incertain",
                        (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "vert=faible std  rouge=forte std (±rel<5%/15%/+)",
                        (8, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (180, 180, 180), 1, cv2.LINE_AA)

        return (_np2t(vis),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 : Road Width Estimator via Metric Depth
# ═══════════════════════════════════════════════════════════════════════════════

class UniDepthRoadWidth:
    """
    Estime la largeur réelle de chaque segment de route en mètres.

    Prend les quad_pairs du node TEST (RoadQuadSegments) — chaque quad est
    un quadrilatère [left_top, right_top, right_bot, left_bot].

    À chaque division le long du segment :
      1. left_pt  = lerp(quad[0], quad[3], t)
      2. right_pt = lerp(quad[1], quad[2], t)
      3. road_dir = direction principale du quad (moy. des deux côtés)
      4. normal   = ⊥ road_dir  (perpendiculaire vraie à la route)
      5. width_px = |dot(right_pt − left_pt, normal)|
      6. depth au milieu → width_m = depth × width_px / fx

    Outputs :
      viz_image      : image annotée avec les mesures par segment
      widths_summary : STRING récapitulatif
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":      ("IMAGE",),
                "quad_pairs": (ROAD_QUAD_PAIRS,),
                "metric_depth": (UNIDEPTH_DEPTH,),
                "intrinsics":   (UNIDEPTH_INTRINSICS,),
            },
            "optional": {
                "n_divisions": ("INT", {
                    "default": 4, "min": 1, "max": 20,
                    "tooltip": "Nombre de mesures de largeur par segment de quad."}),
                "depth_radius": ("INT", {
                    "default": 5, "min": 1, "max": 20,
                    "tooltip": "Rayon du patch (px) pour la médiane de profondeur."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("viz_image", "widths_summary")
    FUNCTION      = "estimate"
    CATEGORY      = "depth/unidepth"

    def estimate(self, image, quad_pairs, metric_depth, intrinsics,
                 n_divisions=4, depth_radius=5):

        rgb  = _t2np_img(image)
        H, W = rgb.shape[:2]

        # Depth → (H, W)
        if metric_depth.dim() == 3:
            depth_np = metric_depth.squeeze(0).cpu().float().numpy()
        else:
            depth_np = metric_depth.cpu().float().numpy()
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # Intrinsics → fx
        K  = intrinsics.cpu().float().numpy()
        fx = float(K[0, 0])

        vis = rgb.copy()
        lines_out = []

        if not quad_pairs:
            cv2.putText(vis, "Aucun quad_pairs reçu — connecter la sortie TEST",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            return (_np2t(vis), "Aucun quad")

        # Dessiner les bords de tous les quads (léger, pour contexte)
        for entry in quad_pairs:
            quad  = entry["quad"].astype(np.float32)
            color = entry.get("color", (200, 200, 200))
            bright = tuple(min(255, int(c * 1.3)) for c in color)
            # Côté gauche : quad[0]→quad[3],  côté droit : quad[1]→quad[2]
            for p0, p1 in [(quad[0], quad[3]), (quad[1], quad[2])]:
                cv2.line(vis,
                         (int(p0[0]), int(p0[1])),
                         (int(p1[0]), int(p1[1])),
                         bright, 1, cv2.LINE_AA)

        # Mesures perpendiculaires
        ts = np.linspace(0.0, 1.0, n_divisions + 2)[1:-1]  # évite les extrémités

        for entry in quad_pairs:
            quad     = entry["quad"].astype(np.float64)
            road_idx = entry["road_idx"]
            seg_idx  = entry["seg_idx"]
            color    = entry.get("color", (200, 200, 200))
            meas_color = (0, 230, 230)   # cyan pour les mesures

            # Direction de route = moyenne des deux côtés
            left_dir  = quad[3] - quad[0]   # vecteur côté gauche
            right_dir = quad[2] - quad[1]   # vecteur côté droit
            road_dir  = (left_dir + right_dir) / 2.0
            norm      = np.linalg.norm(road_dir)
            if norm < 1e-6:
                continue
            road_dir /= norm
            # Normale perpendiculaire (vers la droite de la route)
            normal = np.array([-road_dir[1], road_dir[0]])

            for t in ts:
                left_pt  = quad[0] + t * (quad[3] - quad[0])
                right_pt = quad[1] + t * (quad[2] - quad[1])
                mid_pt   = (left_pt + right_pt) / 2.0

                # Largeur pixel perpendiculaire vraie
                width_px = abs(float(np.dot(right_pt - left_pt, normal)))

                xi, yi = int(np.clip(mid_pt[0], 0, W - 1)), int(np.clip(mid_pt[1], 0, H - 1))
                d, std = _get_depth_at(depth_np, xi, yi, radius=depth_radius)

                # Projeter left_pt et right_pt sur la normale pour
                # trouver les extrémités exactes du segment de mesure
                proj_l = float(np.dot(left_pt,  normal))
                proj_r = float(np.dot(right_pt, normal))
                t_center = float(np.dot(mid_pt, road_dir))
                meas_l = mid_pt + (proj_l - float(np.dot(mid_pt, normal))) * normal
                meas_r = mid_pt + (proj_r - float(np.dot(mid_pt, normal))) * normal

                pl = (int(np.clip(meas_l[0], 0, W-1)), int(np.clip(meas_l[1], 0, H-1)))
                pr = (int(np.clip(meas_r[0], 0, W-1)), int(np.clip(meas_r[1], 0, H-1)))

                cv2.line(vis, pl, pr, meas_color, 1, cv2.LINE_AA)
                cv2.circle(vis, pl, 3, (255, 140,  0), -1)
                cv2.circle(vis, pr, 3, (  0, 140, 255), -1)

                if d > 0.1 and fx > 1.0:
                    width_m   = d * width_px / fx
                    width_std = std * width_px / fx
                    label = f"{width_m:.2f}m"
                    if std > 0.05:
                        label += f"±{width_std:.2f}"
                    lx = int(mid_pt[0]) - 22
                    ly = int(mid_pt[1]) - 4
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(vis, (lx - 1, ly - th - 1), (lx + tw + 1, ly + 1),
                                  (0, 0, 0), -1)
                    cv2.putText(vis, label, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, meas_color, 1, cv2.LINE_AA)
                    lines_out.append(
                        f"R{road_idx+1}-S{seg_idx+1} t={t:.2f}  "
                        f"width_px={width_px:.1f}  depth={d:.2f}m  "
                        f"largeur={width_m:.3f}m ±{width_std:.3f}"
                    )
                else:
                    lines_out.append(
                        f"R{road_idx+1}-S{seg_idx+1} t={t:.2f}  "
                        f"width_px={width_px:.1f}  depth=N/A"
                    )

        cv2.putText(vis, "cyan=mesure perp  orange=bord gauche  bleu=bord droit",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1, cv2.LINE_AA)

        summary = "UniDepth Road Width (perpendicular)\n" + "\n".join(lines_out)
        return (_np2t(vis), summary)


NODE_CLASS_MAPPINGS = {
    "UniDepthPointViz":   UniDepthPointViz,
    "UniDepthRoadWidth":  UniDepthRoadWidth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniDepthPointViz":   "UniDepth Point Visualizer",
    "UniDepthRoadWidth":  "UniDepth Road Width",
}
