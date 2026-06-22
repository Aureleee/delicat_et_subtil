"""
SAM3 All Roads
==============
Détecte TOUTES les routes dans l'image via SAM3 (sans limite max_det=1).
Retourne :
  - colored_viz  : IMAGE avec chaque route dans une couleur différente
  - merged_mask  : MASK union de toutes les routes (pour pipeline existant)
  - individual_masks : MASK batch (N, H, W) — une par route détectée
  - n_roads      : INT nombre de routes trouvées
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import comfy.model_management
import comfy.utils


# Palette de couleurs distinctes (BGR pour cv2, on convertit en RGB)
_PALETTE_RGB = [
    (255,  80,  80),   # rouge
    ( 80, 200,  80),   # vert
    ( 80, 120, 255),   # bleu
    (255, 200,  50),   # jaune
    (200,  80, 255),   # violet
    ( 50, 220, 220),   # cyan
    (255, 140,  50),   # orange
    (180, 255,  80),   # vert clair
    (255,  80, 180),   # rose
    ( 80, 255, 200),   # turquoise
]


def _road_direction(mask_np: np.ndarray) -> np.ndarray:
    """Direction principale d'une route via PCA sur ses pixels."""
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) < 10:
        return np.array([0.0, 1.0])
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt[0]


def _overlap_axis(overlap: np.ndarray) -> np.ndarray:
    """Axe long de la zone de chevauchement via PCA."""
    ys, xs = np.where(overlap)
    if len(xs) < 4:
        return np.array([1.0, 0.0])
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    return Vt[0]


def _resolve_overlaps_directional(masks_np: list) -> list:
    """
    Pour chaque paire (i, j) avec chevauchement :
      - Calcule la direction globale de chaque route (PCA sur tous ses pixels)
      - Calcule l'axe long de la zone d'overlap (PCA sur les pixels communs)
      - La route la plus alignée avec cet axe garde les pixels
      - L'autre perd ces pixels (intersection "saut")
    Ne favorise pas la plus grande — favorise la plus alignée.
    """
    n = len(masks_np)
    cleaned = [m.copy() for m in masks_np]
    directions = [_road_direction(m) for m in masks_np]

    for i in range(n):
        for j in range(i + 1, n):
            overlap = (cleaned[i] > 0.5) & (cleaned[j] > 0.5)
            if overlap.sum() < 20:
                continue
            axis = _overlap_axis(overlap)
            align_i = abs(float(np.dot(directions[i], axis)))
            align_j = abs(float(np.dot(directions[j], axis)))
            if align_i >= align_j:
                cleaned[j][overlap] = 0.0
            else:
                cleaned[i][overlap] = 0.0

    return cleaned


def _refine_mask(sam3_model, orig_image_hwc, coarse_mask, box_xyxy, H, W, device, dtype, iterations=2):
    """Repris de nodes_sam3.py — raffine un masque coarse via le décodeur SAM."""
    def _fallback():
        return (F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0),
                              size=(H, W), mode="bilinear", align_corners=False)[0] > 0).float()
    if iterations <= 0:
        return _fallback()

    pad = 0.1
    x1, y1, x2, y2 = box_xyxy.tolist()
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad))
    cx2, cy2 = min(W, int(x2 + bw * pad)), min(H, int(y2 + bh * pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return _fallback()

    crop = orig_image_hwc[cy1:cy2, cx1:cx2, :3]
    crop_1008 = comfy.utils.common_upscale(crop.unsqueeze(0).movedim(-1, 1),
                                            1008, 1008, "bilinear", crop="disabled")
    crop_frame = crop_1008.to(device=device, dtype=dtype)
    crop_h, crop_w = cy2 - cy1, cx2 - cx1

    mh, mw = coarse_mask.shape[-2:]
    mx1, my1 = int(cx1 / W * mw), int(cy1 / H * mh)
    mx2, my2 = int(cx2 / W * mw), int(cy2 / H * mh)
    if mx2 <= mx1 or my2 <= my1:
        return _fallback()

    mask_logit = coarse_mask[..., my1:my2, mx1:mx2].unsqueeze(0).unsqueeze(0)
    for _ in range(iterations):
        coarse_in = F.interpolate(mask_logit, size=(1008, 1008), mode="bilinear", align_corners=False)
        mask_logit = sam3_model.forward_segment(crop_frame, mask_inputs=coarse_in)

    refined = F.interpolate(mask_logit, size=(crop_h, crop_w), mode="bilinear", align_corners=False)
    full = torch.zeros(1, 1, H, W, device=device, dtype=dtype)
    full[:, :, cy1:cy2, cx1:cx2] = refined
    coarse_full = F.interpolate(coarse_mask.unsqueeze(0).unsqueeze(0),
                                 size=(H, W), mode="bilinear", align_corners=False)
    return ((full[0] > 0) | (coarse_full[0] > 0)).float()


class SAM3AllRoads:
    """
    Détecte TOUTES les routes dans l'image sans la limite max_det=1 du node standard.

    Contrairement à SAM3_Detect qui prend max 1 détection par prompt, ce node
    garde TOUTES les détections au-dessus du threshold et les retourne
    avec des couleurs différentes.

    Outputs :
      colored_viz      : IMAGE avec chaque route colorée différemment
      merged_mask      : MASK union de toutes les routes (blanc = route)
      individual_masks : MASK batch (N, H, W) une par route (pour pipeline)
      n_roads          : INT nombre de routes détectées
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("MODEL",),
                "image":        ("IMAGE",),
                "conditioning": ("CONDITIONING",),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "Score minimum pour garder une détection. "
                               "Baisser pour détecter plus de routes (risque de faux positifs)."}),
                "max_roads": ("INT", {
                    "default": 10, "min": 1, "max": 50,
                    "tooltip": "Nombre maximum de routes à détecter. "
                               "Les routes sont triées par score décroissant."}),
                "refine_iterations": ("INT", {
                    "default": 2, "min": 0, "max": 5,
                    "tooltip": "Passes de raffinement SAM (0 = masques bruts du détecteur)."}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "MASK", "INT")
    RETURN_NAMES  = ("colored_viz", "merged_mask", "individual_masks", "n_roads")
    FUNCTION      = "detect_all"
    CATEGORY      = "road/insertion"

    def detect_all(self, model, image, conditioning, threshold=0.5,
                   max_roads=10, refine_iterations=2):

        B, H, W, C = image.shape
        image_in = comfy.utils.common_upscale(
            image[..., :3].movedim(-1, 1), 1008, 1008, "bilinear", crop="disabled")

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype  = model.model.get_dtype()
        sam3   = model.model.diffusion_model

        # Extraire les embeddings texte du conditioning
        cond_meta = conditioning[0][1]
        multi = cond_meta.get("sam3_multi_cond")
        if multi is not None:
            text_emb  = multi[0]["cond"].to(device=device, dtype=dtype)
            text_mask = multi[0]["attention_mask"]
        else:
            text_emb  = conditioning[0][0].to(device=device, dtype=dtype)
            text_mask = cond_meta.get("attention_mask")

        if text_mask is not None:
            text_mask = text_mask.to(device)
        else:
            text_mask = torch.ones(text_emb.shape[0], text_emb.shape[1],
                                   dtype=torch.int64, device=device)

        # ── Inférence sur le premier frame (batch=1 pour l'instant) ──────────
        frame = image_in[0:1].to(device=device, dtype=dtype)

        results = sam3(
            frame,
            text_embeddings=text_emb,
            text_mask=text_mask,
            boxes=None,
            threshold=threshold,
            orig_size=(H, W),
        )

        pred_boxes = results["boxes"][0]
        scores     = results["scores"][0]
        masks_raw  = results["masks"][0]

        probs = scores.sigmoid()
        keep  = probs > threshold
        kept_boxes  = pred_boxes[keep].cpu()
        kept_scores = probs[keep].cpu()
        kept_masks  = masks_raw[keep]

        # Trier par score et limiter
        order = kept_scores.argsort(descending=True)[:max_roads]
        kept_boxes  = kept_boxes[order]
        kept_scores = kept_scores[order]
        kept_masks  = kept_masks[order]

        n_roads = int(kept_masks.shape[0])

        if n_roads == 0:
            empty_mask = torch.zeros(H, W)
            empty_indiv = torch.zeros(1, H, W)
            rgb = (image[0].cpu().float().numpy() * 255).astype(np.uint8)
            cv2.putText(rgb, f"Aucune route détectée (threshold={threshold})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2)
            viz_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)
            return (viz_t, empty_mask, empty_indiv, 0)

        # ── Raffiner chaque masque ────────────────────────────────────────────
        refined = []
        for m, box in zip(kept_masks, kept_boxes):
            r = _refine_mask(sam3, image[0], m, box, H, W, device, dtype, refine_iterations)
            refined.append(r.squeeze(0).cpu())   # (H, W)

        refined_stack = torch.stack(refined, dim=0)  # (N, H, W)

        # ── Déduplication directionnelle des chevauchements ───────────────────
        # À chaque overlap entre deux routes, la route dont la direction globale
        # est la plus alignée avec l'axe long de l'overlap garde ces pixels.
        # L'autre route perd ces pixels (saut à l'intersection).
        masks_np = [refined[i].numpy() for i in range(n_roads)]
        masks_np = _resolve_overlaps_directional(masks_np)

        # ── Visualisation colorée ─────────────────────────────────────────────
        base_rgb = (image[0].cpu().float().numpy() * 255).astype(np.uint8).copy()
        viz = base_rgb.copy().astype(np.float32)

        for i, (mask_np, score) in enumerate(zip(masks_np, kept_scores.tolist())):
            color = _PALETTE_RGB[i % len(_PALETTE_RGB)]
            mask_bool = mask_np > 0.5
            overlay = np.zeros_like(viz)
            overlay[mask_bool] = color
            viz = np.where(mask_bool[:, :, None], viz * 0.45 + overlay * 0.55, viz)

            mask_u8 = (mask_bool * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz.astype(np.uint8), contours, -1,
                             [int(c * 0.6) for c in color], 2, cv2.LINE_AA)

            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(viz.astype(np.uint8), f"route {i+1}  {score:.2f}",
                                (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        viz = np.clip(viz, 0, 255).astype(np.uint8)

        for i in range(min(n_roads, len(_PALETTE_RGB))):
            color = _PALETTE_RGB[i % len(_PALETTE_RGB)]
            cv2.rectangle(viz, (8, 8 + i * 20), (22, 22 + i * 20), color, -1)
            cv2.putText(viz, f"route {i+1} ({kept_scores[i]:.2f})",
                        (26, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, color, 1, cv2.LINE_AA)

        viz_t = torch.from_numpy(viz.astype(np.float32) / 255.0).unsqueeze(0)

        # ── Outputs ───────────────────────────────────────────────────────────
        clean_stack = torch.stack(
            [torch.from_numpy(m) for m in masks_np], dim=0)  # (N, H, W)
        merged = (clean_stack > 0.5).any(dim=0).float()

        return (viz_t, merged, clean_stack, n_roads)


NODE_CLASS_MAPPINGS        = {"SAM3AllRoads": SAM3AllRoads}
NODE_DISPLAY_NAME_MAPPINGS = {"SAM3AllRoads": "SAM3 All Roads"}
