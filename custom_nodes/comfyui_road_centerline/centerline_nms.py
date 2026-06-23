"""
Node 2 — CenterlineNMS
=======================
NMS Canny-like sur la confidence map (Wei et al. 2020, section 3.4).

Algorithme :
  1. Calcule Dx, Dy (gradients Sobel de la confidence map)
  2. θ = atan2(Dy, Dx)  →  direction perpendiculaire à la route
  3. Pour chaque pixel (x,y) : interpole la confidence en (x±cos θ, y±sin θ)
     Si conf[y,x] est le maximum local dans cette direction → pixel conservé
  4. Seuillage par hysteresis (fort / faible) → centerline binaire finale

Sortie : MASK binaire des centerlines fines
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import map_coordinates, maximum_filter


class CenterlineNMS:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = ("MASK", "IMAGE")
    RETURN_NAMES  = ("centerline_mask", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confidence_map": ("MASK",),
            },
            "optional": {
                "road_mask": ("MASK",),
                "threshold_high": ("FLOAT", {
                    "default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Seuil fort pour l'hysteresis Canny."}),
                "threshold_low": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Seuil faible pour l'hysteresis Canny."}),
                "background_image": ("IMAGE",),
            }
        }

    def run(self, confidence_map, road_mask=None, threshold_high=0.3,
            threshold_low=0.1, background_image=None):

        if confidence_map.dim() == 3:
            conf = confidence_map[0].cpu().numpy().astype(np.float64)
        else:
            conf = confidence_map.cpu().numpy().astype(np.float64)
        H, W = conf.shape

        road_np = None
        if road_mask is not None:
            rm = road_mask[0] if road_mask.dim() == 3 else road_mask
            road_np = (rm.cpu().numpy() > 0.5)

        # ── Gradient (direction perpendiculaire à la route) ────────────────────
        conf_f = conf.astype(np.float32)
        Dx = cv2.Sobel(conf_f, cv2.CV_64F, 1, 0, ksize=3)
        Dy = cv2.Sobel(conf_f, cv2.CV_64F, 0, 1, ksize=3)
        theta = np.arctan2(Dy, Dx)   # direction du gradient = perp à la route

        # ── NMS vectorisé (interpolation bilinéaire) ──────────────────────────
        ys, xs = np.mgrid[0:H, 0:W]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Voisins +1 et -1 dans la direction du gradient
        v_pos = map_coordinates(conf, [ys + sin_t, xs + cos_t],
                                order=1, mode='constant', cval=0.0)
        v_neg = map_coordinates(conf, [ys - sin_t, xs - cos_t],
                                order=1, mode='constant', cval=0.0)

        # Pixel conservé s'il est maximum local ET au-dessus du seuil faible
        is_max  = (conf >= v_pos - 1e-9) & (conf >= v_neg - 1e-9)
        nms_map = np.where(is_max, conf, 0.0).astype(np.float32)

        # Restreindre au mask de route si fourni
        if road_np is not None:
            nms_map *= road_np.astype(np.float32)

        # ── Hysteresis ────────────────────────────────────────────────────────
        strong = (nms_map >= threshold_high).astype(np.uint8)
        weak   = ((nms_map >= threshold_low) & (nms_map < threshold_high)).astype(np.uint8)

        # Propager les pixels forts aux pixels faibles connectés
        # (8-connexité, même principe que Canny)
        kernel = np.ones((3, 3), np.uint8)
        strong_dilated = cv2.dilate(strong, kernel, iterations=1)
        connected_weak = (weak & strong_dilated).astype(np.uint8)
        # Répéter jusqu'à stabilité (max 10 itérations)
        for _ in range(10):
            prev = connected_weak.copy()
            dilated = cv2.dilate((strong | connected_weak).astype(np.uint8), kernel, iterations=1)
            connected_weak = (weak & dilated).astype(np.uint8)
            if np.array_equal(connected_weak, prev):
                break

        centerline = ((strong | connected_weak) > 0).astype(np.uint8)

        # ── Debug image ───────────────────────────────────────────────────────
        if background_image is not None:
            bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), dtype=np.uint8)

        # NMS map en overlay chaud
        nms_vis = (nms_map / (nms_map.max() + 1e-9) * 180).astype(np.uint8)
        heat = cv2.applyColorMap(nms_vis, cv2.COLORMAP_AUTUMN)
        mask_nms = (nms_map > threshold_low).astype(np.uint8)
        debug = cv2.addWeighted(debug, 0.6, heat * mask_nms[:, :, None], 0.4, 0)
        # Centerline en cyan
        debug[centerline > 0] = [255, 255, 0]   # BGR jaune-vert

        n_px = int(centerline.sum())
        cv2.putText(debug, f"centerline: {n_px}px",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        center_t = torch.from_numpy(centerline.astype(np.float32)).unsqueeze(0)
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        debug_t   = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (center_t, debug_t)


NODE_CLASS_MAPPINGS        = {"CenterlineNMS": CenterlineNMS}
NODE_DISPLAY_NAME_MAPPINGS = {"CenterlineNMS": "Road CenterlineNMS"}
