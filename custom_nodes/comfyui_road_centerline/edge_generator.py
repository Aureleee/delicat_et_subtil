"""
Node 4 — EdgeLineGenerator
============================
Génère les lignes de bord de route (Wei et al. 2020, section 3.4).

Pour chaque point p de la centerline trackée :
  - Direction tangente θ = angles[i]
  - Vecteur perpendiculaire : perp = (-sin θ, cos θ)
  - Largeur locale w = widths[i]
  - Edge gauche  = p + (w/2) * perp
  - Edge droite  = p - (w/2) * perp

Sortie : MASK des edges (binaire) + IMAGE debug
"""

import numpy as np
import cv2
import torch

ROAD_PATHS = "ROAD_PATHS"


def _smooth_angles(angles, window=5):
    """Lissage circulaire des angles (évite les sauts ±π)."""
    n = len(angles)
    if n <= window:
        return angles
    smoothed = np.array(angles, dtype=np.float64)
    dx = np.cos(smoothed)
    dy = np.sin(smoothed)
    kernel = np.ones(window) / window
    dx_s = np.convolve(dx, kernel, mode='same')
    dy_s = np.convolve(dy, kernel, mode='same')
    return np.arctan2(dy_s, dx_s)


class EdgeLineGenerator:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES  = ("left_edge_mask", "right_edge_mask", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "road_paths": (ROAD_PATHS,),
                "width_map":  ("MASK",),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "width_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Multiplie la largeur estimée. >1 pour élargir les edges."}),
                "edge_thickness": ("INT", {
                    "default": 2, "min": 1, "max": 8,
                    "tooltip": "Épaisseur du tracé des edge lines en pixels."}),
                "smooth_window": ("INT", {
                    "default": 7, "min": 1, "max": 25,
                    "tooltip": "Fenêtre de lissage des angles (évite les zigzags)."}),
            }
        }

    def run(self, road_paths, width_map,
            background_image=None, width_scale=1.0,
            edge_thickness=2, smooth_window=7):

        wm = width_map[0] if width_map.dim() == 3 else width_map
        wm_np = wm.cpu().numpy().astype(np.float32)
        H, W = wm_np.shape

        left_mask  = np.zeros((H, W), dtype=np.uint8)
        right_mask = np.zeros((H, W), dtype=np.uint8)

        if background_image is not None:
            bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), dtype=np.uint8)

        for path in (road_paths or []):
            pts    = path["points"]   # (N,2) xy
            widths = path["widths"]   # (N,)
            angles = path["angles"]   # (N,) radians

            if len(pts) < 2:
                continue

            # Lissage des angles
            angles_s = _smooth_angles(angles, window=smooth_window)

            left_pts  = []
            right_pts = []

            for i in range(len(pts)):
                x, y = float(pts[i, 0]), float(pts[i, 1])
                # Largeur : utiliser width_map local si disponible
                xi, yi = int(round(x)), int(round(y))
                xi = np.clip(xi, 0, W - 1); yi = np.clip(yi, 0, H - 1)
                local_w = float(wm_np[yi, xi])
                if local_w < 1.0:
                    local_w = float(widths[i])
                half_w = (local_w / 2.0) * width_scale

                theta = float(angles_s[i])
                # Perpendiculaire : (-sin θ, cos θ)
                px = -np.sin(theta)
                py =  np.cos(theta)

                lx = x + px * half_w; ly = y + py * half_w
                rx = x - px * half_w; ry = y - py * half_w

                left_pts.append((int(round(lx)), int(round(ly))))
                right_pts.append((int(round(rx)), int(round(ry))))

            # Dessiner les polylignes sur les masks et le debug
            def draw_poly(pts_list, mask, color_bgr):
                arr = np.array(pts_list, dtype=np.int32)
                arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
                arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
                for a, b in zip(arr[:-1], arr[1:]):
                    cv2.line(mask,  (a[0], a[1]), (b[0], b[1]), 1,         edge_thickness)
                    cv2.line(debug, (a[0], a[1]), (b[0], b[1]), color_bgr, edge_thickness, cv2.LINE_AA)

            draw_poly(left_pts,  left_mask,  (0, 80, 255))    # orange gauche
            draw_poly(right_pts, right_mask, (255, 80, 0))    # bleu droite

            # Centerline en blanc sur debug
            pts_arr = pts.astype(np.int32)
            pts_arr[:, 0] = np.clip(pts_arr[:, 0], 0, W - 1)
            pts_arr[:, 1] = np.clip(pts_arr[:, 1], 0, H - 1)
            for a, b in zip(pts_arr[:-1], pts_arr[1:]):
                cv2.line(debug, (a[0], a[1]), (b[0], b[1]), (200, 200, 200), 1, cv2.LINE_AA)

        cv2.putText(debug, f"edges: {int(left_mask.sum())}+{int(right_mask.sum())} px",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        left_t   = torch.from_numpy(left_mask.astype(np.float32)).unsqueeze(0)
        right_t  = torch.from_numpy(right_mask.astype(np.float32)).unsqueeze(0)
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        debug_t   = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (left_t, right_t, debug_t)


NODE_CLASS_MAPPINGS        = {"EdgeLineGenerator": EdgeLineGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"EdgeLineGenerator": "Road EdgeGen"}
