"""
Node 5 — TensorVoting
======================
Détection des intersections de route par tensor voting (Wei et al. 2020, section 3.5).

Algorithme (Eq. 13-15 du papier) :
  Pour chaque pixel p de la région route :
    n = vecteur normal à p  (gradient du SDF / distance transform)
    T_p = n * nᵀ   (tenseur stick 2×2)

  Vote de P vers O via une osculating circle decay :
    DF = exp(-(s² + c*κ²/ε²))
    où s = longueur d'arc, κ = courbure, ε = scale, c = constante

  Après voting : saliency ball = λ_min du tenseur voté
    → haute aux intersections (directions multiples convergent)
    → basse sur les segments droits (un seul stick dominant)

  NMS sur ball saliency → intersection points

Implémentation :
  On vectorise le voting par approximation convolutive :
  les composantes du tenseur (Txx, Txy, Tyy) sont lissées par un filtre gaussien
  (équivalent au voting avec DF gaussienne), puis on calcule les eigenvalues.
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import gaussian_filter, distance_transform_edt, maximum_filter
from skimage.morphology import skeletonize


class TensorVoting:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES  = ("ball_saliency", "intersection_mask", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "road_mask": ("MASK",),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "voting_sigma": ("FLOAT", {
                    "default": 8.0, "min": 1.0, "max": 40.0, "step": 0.5,
                    "tooltip": "ε : portée du vote (≈ rayon d'influence en pixels). "
                               "Papier : calibré sur la largeur de route."}),
                "ball_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Seuil sur la saliency ball normalisée [0,1] pour détecter une intersection."}),
                "nms_radius": ("INT", {
                    "default": 12, "min": 3, "max": 40,
                    "tooltip": "Rayon de NMS autour de chaque intersection détectée (pixels)."}),
                "min_road_area": ("INT", {
                    "default": 200, "min": 0, "max": 5000}),
            }
        }

    def run(self, road_mask, background_image=None,
            voting_sigma=8.0, ball_threshold=0.35,
            nms_radius=12, min_road_area=200):

        rm = road_mask[0] if road_mask.dim() == 3 else road_mask
        mask_np = (rm.cpu().numpy() > 0.5).astype(np.uint8)
        H, W = mask_np.shape

        # Nettoyage
        if min_road_area > 0:
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
            clean = np.zeros_like(mask_np)
            for i in range(1, n_cc):
                if stats[i, cv2.CC_STAT_AREA] >= min_road_area:
                    clean[labels == i] = 1
            mask_np = clean

        # ── Encodage des tenseurs stick ────────────────────────────────────────
        # Gradient du SDF donne les normales intérieures à la route
        dist = distance_transform_edt(mask_np).astype(np.float32)
        # Normaliser par max pour avoir un SDF [0,1]
        sdf = dist / (dist.max() + 1e-9)

        # Gradients (direction normale à la route)
        Dx = cv2.Sobel(sdf, cv2.CV_64F, 1, 0, ksize=3)
        Dy = cv2.Sobel(sdf, cv2.CV_64F, 0, 1, ksize=3)
        norm_g = np.sqrt(Dx ** 2 + Dy ** 2) + 1e-12
        nx = (Dx / norm_g).astype(np.float32)
        ny = (Dy / norm_g).astype(np.float32)

        # Tenseur stick T = n*nᵀ (seulement sur les pixels route)
        road_f = mask_np.astype(np.float32)
        Txx = (nx * nx) * road_f   # composante (0,0)
        Txy = (nx * ny) * road_f   # composante (0,1) = (1,0)
        Tyy = (ny * ny) * road_f   # composante (1,1)

        # ── Voting par lissage gaussien (approximation convolutive) ───────────
        # DF gaussienne : DF(d) = exp(-d²/2σ²) → convolution avec G_σ
        Vxx = gaussian_filter(Txx, sigma=voting_sigma)
        Vxy = gaussian_filter(Txy, sigma=voting_sigma)
        Vyy = gaussian_filter(Tyy, sigma=voting_sigma)

        # Restreindre au mask route
        Vxx *= road_f; Vxy *= road_f; Vyy *= road_f

        # ── Eigenvalues du tenseur voté ────────────────────────────────────────
        # [[a, b], [b, c]] : λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
        a = Vxx; b = Vxy; c = Vyy
        half_trace = (a + c) * 0.5
        disc       = np.sqrt(np.maximum(((a - c) * 0.5) ** 2 + b ** 2, 0.0))
        lambda1    = half_trace + disc   # eigenvalue dominante (stick)
        lambda2    = half_trace - disc   # eigenvalue minoritaire (ball saliency)

        # Ball saliency = λ_min normalisé
        ball = np.clip(lambda2, 0.0, None).astype(np.float32)
        ball *= road_f
        ball_max = ball.max()
        if ball_max > 1e-9:
            ball /= ball_max

        # ── NMS sur ball saliency ─────────────────────────────────────────────
        ball_dilated = maximum_filter(ball, size=nms_radius * 2 + 1)
        local_max    = (ball >= ball_dilated - 1e-6) & (ball >= ball_threshold)
        inter_mask   = (local_max & (road_f > 0)).astype(np.uint8)

        # ── Debug image ───────────────────────────────────────────────────────
        if background_image is not None:
            bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), dtype=np.uint8)
            debug[mask_np > 0] = [40, 60, 40]

        # Overlay ball saliency (heatmap)
        ball_u8 = (ball * 255).astype(np.uint8)
        heat = cv2.applyColorMap(ball_u8, cv2.COLORMAP_JET)
        mask_road_3 = np.stack([road_f, road_f, road_f], axis=2).astype(np.uint8)
        heat_masked = heat * mask_road_3
        debug = cv2.addWeighted(debug, 0.5, heat_masked, 0.5, 0)

        # Intersections en rouge vif avec cercle
        inter_pts = np.argwhere(inter_mask > 0)
        for (iy, ix) in inter_pts:
            cv2.circle(debug, (int(ix), int(iy)), nms_radius // 2, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(debug, (int(ix), int(iy)), 3, (0, 255, 255), -1)

        cv2.putText(debug, f"{len(inter_pts)} intersections",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        ball_t  = torch.from_numpy(ball).unsqueeze(0)
        inter_t = torch.from_numpy(inter_mask.astype(np.float32)).unsqueeze(0)
        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        debug_t   = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (ball_t, inter_t, debug_t)


NODE_CLASS_MAPPINGS        = {"TensorVoting": TensorVoting}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorVoting": "Road TensorVoting"}
