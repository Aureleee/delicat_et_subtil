"""
Node 1 — MaskToConfidenceWidth
================================
Remplace le CNN du papier (Wei et al. 2020).
A partir du mask de route binaire, génère :
  - confidence_map : carte gaussienne centrée sur la centerline
  - width_map      : largeur de route (diamètre) en pixels en chaque point

Mathématiques :
  dist = distance_transform_edt(mask)   →  demi-largeur au bord le plus proche
  width_map = 2 * dist
  skeleton = skeletonize(mask)
  confidence_map = GaussianFilter(skeleton * dist, σ)   normalisé [0,1]
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize


class MaskToConfidenceWidth:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES  = ("confidence_map", "width_map", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "road_mask": ("MASK",),
            },
            "optional": {
                "conf_sigma_factor": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 1.5, "step": 0.05,
                    "tooltip": "σ du filtre gaussien = facteur × demi-largeur médiane de route. "
                               "Plus grand → confidence map plus lisse."}),
                "min_road_area_px": ("INT", {
                    "default": 300, "min": 0, "max": 10000,
                    "tooltip": "Supprime les petites régions de route (bruit)."}),
            }
        }

    def run(self, road_mask, conf_sigma_factor=0.3, min_road_area_px=300):
        if road_mask.dim() == 3:
            mask_np = (road_mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        else:
            mask_np = (road_mask.cpu().numpy() > 0.5).astype(np.uint8)

        H, W = mask_np.shape

        # Nettoyage : suppression petites composantes
        if min_road_area_px > 0:
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            clean = np.zeros_like(mask_np)
            for i in range(1, n_cc):
                if stats[i, cv2.CC_STAT_AREA] >= min_road_area_px:
                    clean[labels == i] = 1
            mask_np = clean

        # Distance transform → demi-largeur en chaque pixel
        dist = distance_transform_edt(mask_np).astype(np.float32)

        # Width map = diamètre
        width_map = (dist * 2.0).astype(np.float32)

        # Squelette (centerline)
        skel = skeletonize(mask_np > 0).astype(np.float32)

        # Confidence map : squelette pondéré par la distance locale → lissage gaussien
        road_dists = dist[mask_np > 0]
        median_half_w = float(np.median(road_dists)) if len(road_dists) > 0 else 3.0
        sigma = max(1.0, median_half_w * conf_sigma_factor)

        conf_map = gaussian_filter(skel * dist, sigma=sigma).astype(np.float32)
        cmax = conf_map.max()
        if cmax > 1e-9:
            conf_map /= cmax

        # Debug image
        debug = np.zeros((H, W, 3), dtype=np.uint8)
        # Route en vert foncé
        debug[mask_np > 0] = [30, 80, 30]
        # Width map : canal bleu (normalisé)
        wmax = width_map.max()
        if wmax > 0:
            w_vis = (dist / (wmax / 2.0) * 120).clip(0, 200).astype(np.uint8)
            debug[:, :, 0] = (w_vis * (mask_np > 0)).astype(np.uint8)
        # Squelette en jaune
        debug[skel > 0] = [255, 255, 0]
        # Confidence en rouge (overlay)
        conf_vis = (conf_map * 255).astype(np.uint8)
        debug[:, :, 2] = np.maximum(debug[:, :, 2], conf_vis)

        conf_t  = torch.from_numpy(conf_map).unsqueeze(0)
        width_t = torch.from_numpy(width_map).unsqueeze(0)
        debug_t = torch.from_numpy(debug.astype(np.float32) / 255.0).unsqueeze(0)

        return (conf_t, width_t, debug_t)


NODE_CLASS_MAPPINGS        = {"MaskToConfidenceWidth": MaskToConfidenceWidth}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskToConfidenceWidth": "Road MaskToMaps"}
