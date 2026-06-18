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
    Estime la largeur réelle de la route en mètres en utilisant :
    - Les bords de route (LINE_SEGMENTS de RoadMaskBorderFit)
    - La metric depth (UNIDEPTH_DEPTH) pour convertir pixels → mètres
    - Les intrinsics caméra (UNIDEPTH_INTRINSICS) pour la conversion angulaire

    Principe :
      À chaque niveau Y (plusieurs hauteurs dans l'image), on intersecte les
      deux droites de bord avec la ligne horizontale, on calcule l'angle angulaire
      entre les deux points, et on utilise la depth pour convertir en distance réelle :
        largeur_m = depth × tan(angle/2) × 2  ≈  depth × (Δx_px / fx)

    Outputs :
      viz_image   : image avec bords tracés + largeurs annotées
      widths_str  : STRING avec tableau des largeurs par niveau Y
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "border_lines": ("LINE_SEGMENTS",),
                "metric_depth": (UNIDEPTH_DEPTH,),
                "intrinsics":   (UNIDEPTH_INTRINSICS,),
            },
            "optional": {
                "road_mask": ("IMAGE", {
                    "tooltip": "Masque de route optionnel (pour filtrer les points hors route)."}),
                "n_levels": ("INT", {
                    "default": 6, "min": 2, "max": 20,
                    "tooltip": "Nombre de niveaux Y où estimer la largeur."}),
                "margin_frac": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.4, "step": 0.01}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("viz_image", "widths_summary")
    FUNCTION      = "estimate"
    CATEGORY      = "depth/unidepth"

    def estimate(self, image, border_lines, metric_depth, intrinsics,
                 road_mask=None, n_levels=6, margin_frac=0.1):

        rgb   = _t2np_img(image)
        H, W  = rgb.shape[:2]

        # Depth
        if metric_depth.dim() == 3:
            depth_np = metric_depth.squeeze(0).cpu().float().numpy()
        else:
            depth_np = metric_depth.cpu().float().numpy()
        if depth_np.shape != (H, W):
            depth_np = cv2.resize(depth_np, (W, H), interpolation=cv2.INTER_LINEAR)

        # Intrinsics
        K = intrinsics.cpu().float().numpy()
        fx = float(K[0, 0])

        # Segments de bord
        lines   = border_lines.get("lines", np.zeros((0, 2, 2)))
        bW, bH  = border_lines.get("width", W), border_lines.get("height", H)

        vis = rgb.copy()

        if lines.shape[0] < 2:
            cv2.putText(vis, "Besoin de 2 segments de bord (gauche + droite)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
            return (_np2t(vis), "Pas assez de segments")

        seg_l, seg_r = lines[0], lines[1]

        # ── Normale perpendiculaire à la route ────────────────────────────────
        # Direction moyenne des deux bords → normale = rotation 90°
        def seg_dir_norm(seg):
            d = seg[1] - seg[0]
            n = np.linalg.norm(d)
            return d / n if n > 1e-6 else d

        dir_l = seg_dir_norm(seg_l)
        dir_r = seg_dir_norm(seg_r)
        if np.dot(dir_l, dir_r) < 0:
            dir_r = -dir_r
        road_dir = (dir_l + dir_r) / 2.0
        road_dir /= np.linalg.norm(road_dir) + 1e-9
        # Normale à la route (perpendiculaire, pointant vers la droite)
        road_perp = np.array([road_dir[1], -road_dir[0]])  # rotation -90°
        if road_perp[0] < 0:
            road_perp = -road_perp  # toujours vers x positif

        # Tracer les bords
        def draw_seg(s, color, label):
            p0 = tuple(np.round(s[0]).astype(int))
            p1 = tuple(np.round(s[1]).astype(int))
            cv2.line(vis, p0, p1, color, 3, cv2.LINE_AA)
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            cv2.putText(vis, label, (mid[0] + 5, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        draw_seg(seg_l, (255, 140, 0), "gauche")
        draw_seg(seg_r, (0, 160, 255), "droite")

        # ── x(y) pour un segment linéaire ────────────────────────────────────
        def seg_x_at_y(seg, y):
            """Interpoler x sur le segment à hauteur y."""
            x0, y0 = seg[0]
            x1, y1 = seg[1]
            if abs(y1 - y0) < 1e-3:
                return (x0 + x1) / 2.0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)

        # Y-range commun aux deux segments
        yl0 = max(min(seg_l[0, 1], seg_l[1, 1]), min(seg_r[0, 1], seg_r[1, 1]))
        yl1 = min(max(seg_l[0, 1], seg_l[1, 1]), max(seg_r[0, 1], seg_r[1, 1]))
        my  = int(margin_frac * (yl1 - yl0))
        yl0 = int(yl0) + my
        yl1 = int(yl1) - my
        if yl1 <= yl0:
            yl0, yl1 = int(yl0), int(yl1) + 20

        ys = np.linspace(yl0, yl1, n_levels).astype(int)

        lines_out = []
        for y in ys:
            y = int(np.clip(y, 0, H - 1))
            xl = float(seg_x_at_y(seg_l, y))
            xr = float(seg_x_at_y(seg_r, y))
            if xl > xr:
                xl, xr = xr, xl

            xli, xri = int(np.clip(xl, 0, W - 1)), int(np.clip(xr, 0, W - 1))

            # Depth au centre de la route à ce niveau
            xc = (xli + xri) // 2
            d, std = _get_depth_at(depth_np, xc, y, radius=5)

            if d < 0.1 or fx < 1.0:
                lines_out.append(f"y={y:4d}  xl={xl:.0f} xr={xr:.0f}  depth=N/A")
                continue

            # Vecteur pixel entre les deux bords (horizontal à ce niveau y)
            delta_px = np.array([xr - xl, 0.0])
            # Projection sur la normale à la route = vraie largeur perpendiculaire
            perp_px = abs(float(np.dot(delta_px, road_perp)))

            width_m = d * perp_px / fx
            width_std = std * perp_px / fx

            # Dessiner la ligne de mesure perpendiculaire (pas horizontale)
            # Point gauche et droite sur la perpendiculaire passant par le centre
            half = road_perp * perp_px / 2.0
            pl = (int(xc - half[0]), int(y - half[1]))
            pr = (int(xc + half[0]), int(y + half[1]))
            cv2.line(vis, pl, pr, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, (xli, y), 4, (255, 140, 0), -1, cv2.LINE_AA)
            cv2.circle(vis, (xri, y), 4, (0, 160, 255), -1, cv2.LINE_AA)

            label = f"{width_m:.2f}m"
            if std > 0.05:
                label += f" ±{width_std:.2f}"
            cv2.putText(vis, label, (xc - 20, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            lines_out.append(
                f"y={y:4d}  xl={xl:6.1f} xr={xr:6.1f}  "
                f"perp={perp_px:5.1f}px  depth={d:.2f}m  largeur⊥={width_m:.2f}m ±{width_std:.2f}"
            )

        # Légende
        cv2.putText(vis, "cyan=mesure  orange=gauche  bleu=droite",
                    (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (180, 180, 180), 1, cv2.LINE_AA)

        summary = "UniDepth Road Width Estimator\n" + "\n".join(lines_out)
        return (_np2t(vis), summary)


NODE_CLASS_MAPPINGS = {
    "UniDepthPointViz":   UniDepthPointViz,
    "UniDepthRoadWidth":  UniDepthRoadWidth,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UniDepthPointViz":   "UniDepth Point Visualizer",
    "UniDepthRoadWidth":  "UniDepth Road Width",
}
