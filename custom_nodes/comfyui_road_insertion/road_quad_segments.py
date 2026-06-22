"""
TEST — Road Quad Segments
=========================
Route DROITE  : régression linéaire globale sur tous les fragments agrégés.
Route COURBÉE : Douglas-Peucker point-par-point sur la composante principale.
               Les sous-segments localement droits bénéficient aussi du line fit + offset.

Deux sorties debug :
  debug_image  : quads colorés superposés sur le fond (classique)
  masked_debug : fill du quad UNIQUEMENT là où le masque confirme la route,
                 mais les lignes de bord s'étendent sur toute la longueur estimée
"""

import numpy as np
import torch
import cv2


ROAD_QUAD_PAIRS = "ROAD_QUAD_PAIRS"

_ROAD_COLORS = [
    (255, 100, 100),
    (100, 220, 100),
    (100, 140, 255),
    (255, 220,  80),
    (220, 100, 255),
    ( 60, 210, 210),
    (255, 150,  60),
    (180, 255,  80),
]


# ── Overlap resolution ─────────────────────────────────────────────────────────

def _resolve_overlap_pairwise(masks_np):
    n = len(masks_np)
    cleaned = [m.copy() for m in masks_np]

    def _dir(m):
        ys, xs = np.where(m > 0.5)
        if len(xs) < 10:
            return np.array([1.0, 0.0])
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        pts -= pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts, full_matrices=False)
        return Vt[0]

    directions = [_dir(m) for m in masks_np]
    for i in range(n):
        for j in range(i + 1, n):
            overlap = (cleaned[i] > 0.5) & (cleaned[j] > 0.5)
            if overlap.sum() < 20:
                continue
            ys, xs = np.where(overlap)
            pts = np.stack([xs, ys], axis=1).astype(np.float64)
            pts -= pts.mean(axis=0)
            _, _, Vt = np.linalg.svd(pts, full_matrices=False)
            axis = Vt[0]
            if abs(np.dot(directions[i], axis)) >= abs(np.dot(directions[j], axis)):
                cleaned[j][overlap] = 0.0
            else:
                cleaned[i][overlap] = 0.0
    return cleaned


# ── PCA helpers ────────────────────────────────────────────────────────────────

def _pca_straightness(mask_np):
    ys, xs = np.where(mask_np > 0.5)
    if len(xs) < 10:
        return 1.0, np.array([1.0, 0.0]), np.array([0.0, 0.0])
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    center = pts.mean(axis=0)
    _, s, Vt = np.linalg.svd(pts - center, full_matrices=False)
    return float(s[1] / (s[0] + 1e-9)), Vt[0], center


# ── Line fitting ───────────────────────────────────────────────────────────────

def _fit_line_side(pts, n_out, trim=0.05):
    pts = pts.astype(np.float32)
    n = len(pts)
    k = max(1, int(n * trim))
    pts_fit = pts[k:n - k] if n - 2 * k >= 6 else pts
    line = cv2.fitLine(pts_fit.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = float(line[0]), float(line[1]), float(line[2]), float(line[3])
    direction = np.array([vx, vy])
    point     = np.array([x0, y0])
    projs = pts @ direction - float(np.dot(point, direction))
    t_min, t_max = float(projs.min()), float(projs.max())
    ts = np.linspace(t_min, t_max, n_out)
    return (point + np.outer(ts, direction)).astype(np.float32), direction, point


def _offset_side(pts, perp, offset_px):
    return (pts + perp * offset_px).astype(np.float32)


def _perp_from_dir(dl, dr):
    d_avg = (dl + dr) / 2.0
    perp = np.array([-d_avg[1], d_avg[0]], dtype=np.float32)
    n = np.linalg.norm(perp)
    return perp / (n + 1e-9)


def _quads_from_sides(left_r, right_r, max_sub):
    n = len(left_r)
    sub = max(1, min(max_sub, n - 1))
    idxs = np.linspace(0, n - 1, sub + 1, dtype=int)
    return [
        np.array([left_r[idxs[j]], right_r[idxs[j]],
                  right_r[idxs[j+1]], left_r[idxs[j+1]]], dtype=np.float32)
        for j in range(sub)
    ]


# ── Border points (for straight roads with fragments) ─────────────────────────

def _border_points_all_fragments(mask_np, min_area):
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_left, all_right = [], []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area or len(cnt) < 6:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float32)
        top_idx = int(np.argmin(pts[:, 1]))
        bot_idx = int(np.argmax(pts[:, 1]))

        def _arc(a, b):
            if a <= b: return pts[a:b + 1]
            return np.concatenate([pts[a:], pts[:b + 1]], axis=0)

        side_a = _arc(top_idx, bot_idx)
        side_b = _arc(bot_idx, top_idx)[::-1]
        if np.median(side_a[:, 0]) <= np.median(side_b[:, 0]):
            all_left.append(side_a); all_right.append(side_b)
        else:
            all_left.append(side_b); all_right.append(side_a)

    if not all_left:
        return None, None
    return np.concatenate(all_left, axis=0), np.concatenate(all_right, axis=0)


# ── Resample + DP ─────────────────────────────────────────────────────────────

def _resample(pts, n):
    if len(pts) < 2:
        return np.tile(pts[0:1], (n, 1))
    diffs  = np.diff(pts, axis=0)
    cumlen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    total  = cumlen[-1]
    if total < 1e-6:
        return np.tile(pts[0:1], (n, 1))
    t = np.linspace(0.0, total, n)
    return np.stack([np.interp(t, cumlen, pts[:, k]) for k in range(2)], axis=1)


def _dp_indices(pts, epsilon):
    if len(pts) < 3:
        return np.arange(len(pts))
    approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                               epsilon, closed=False).reshape(-1, 2)
    indices, start = [], 0
    for ap in approx:
        idx = int(np.argmin(np.linalg.norm(pts[start:] - ap, axis=1))) + start
        indices.append(idx)
        start = max(start, idx)
    return np.array(sorted(set(indices)), dtype=int)


def _merge_kp(idx_l, idx_r, n, min_gap):
    all_kp = sorted(set(idx_l.tolist()) | set(idx_r.tolist()))
    deduped = [all_kp[0]]
    for k in all_kp[1:]:
        if k - deduped[-1] >= min_gap:
            deduped.append(k)
    if deduped[0] != 0: deduped.insert(0, 0)
    if deduped[-1] != n - 1: deduped.append(n - 1)
    return deduped


def _segment_straightness(left_r, right_r, i0, i1):
    centers = ((left_r[i0:i1+1] + right_r[i0:i1+1]) / 2.0).astype(np.float64)
    if len(centers) < 4:
        return 0.0
    c = centers - centers.mean(axis=0)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    return float(s[1] / (s[0] + 1e-9))


# ── Curved road processor ─────────────────────────────────────────────────────

def _process_curved(mask_np, sample_pts, dp_epsilon, max_sub, min_gap,
                    min_area, straight_threshold, border_offset_px):
    """
    Composante principale (la plus grande) : DP + line fit par segment droit.
    """
    mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area and len(c) >= 6]
    if not valid:
        return [], [], []

    pts = max(valid, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    top_idx = int(np.argmin(pts[:, 1]))
    bot_idx = int(np.argmax(pts[:, 1]))

    def _arc(a, b):
        if a <= b: return pts[a:b + 1]
        return np.concatenate([pts[a:], pts[:b + 1]], axis=0)

    side_a = _arc(top_idx, bot_idx)
    side_b = _arc(bot_idx, top_idx)[::-1]
    if np.median(side_a[:, 0]) <= np.median(side_b[:, 0]):
        left, right = side_a, side_b
    else:
        left, right = side_b, side_a

    left_r  = _resample(left,  sample_pts)
    right_r = _resample(right, sample_pts)

    idx_l = _dp_indices(left_r,  dp_epsilon)
    idx_r = _dp_indices(right_r, dp_epsilon)
    kps   = _merge_kp(idx_l, idx_r, sample_pts, min_gap)

    left_kps  = [left_r[ki].astype(int) for ki in idx_l[1:-1]]
    right_kps = [right_r[ki].astype(int) for ki in idx_r[1:-1]]

    quads = []
    for i in range(len(kps) - 1):
        i0, i1 = kps[i], kps[i+1]
        if i1 <= i0:
            continue
        seg_ratio = _segment_straightness(left_r, right_r, i0, i1)
        if seg_ratio < straight_threshold:
            sl, dl, _ = _fit_line_side(left_r[i0:i1+1],  max_sub + 1)
            sr, dr, _ = _fit_line_side(right_r[i0:i1+1], max_sub + 1)
            if border_offset_px != 0:
                perp = _perp_from_dir(dl, dr)
                sl = _offset_side(sl, -perp, border_offset_px)
                sr = _offset_side(sr,  perp, border_offset_px)
            quads.extend(_quads_from_sides(sl, sr, max_sub))
        else:
            sub  = max(1, min(max_sub, i1 - i0))
            idxs = np.linspace(i0, i1, sub + 1, dtype=int)
            for j in range(sub):
                quads.append(np.array([
                    left_r[idxs[j]],    right_r[idxs[j]],
                    right_r[idxs[j+1]], left_r[idxs[j+1]]
                ], dtype=np.float32))

    return quads, left_kps, right_kps


# ── Hierarchical curved processing ────────────────────────────────────────────

def _quads_to_coverage(quads, H, W):
    """Masque binaire des pixels couverts par l'ensemble des quads."""
    cov = np.zeros((H, W), dtype=np.uint8)
    for quad in quads:
        cv2.fillPoly(cov, [quad.astype(np.int32)], 255)
    return cov


def _process_curved_hierarchical(mask_np, n_passes, sample_pts, dp_epsilon,
                                  max_sub, min_gap, min_area,
                                  straight_threshold, border_offset_px):
    """
    Passe hiérarchique :
      Pass 1 → quads sur la composante principale → masque résiduel
      Pass 2 → même algo sur le résiduel → nouveaux quads
      ...
    Chaque passe "explique" une portion de la route, la retire,
    et laisse la suivante trouver ce qui reste.
    """
    H, W = mask_np.shape
    current = mask_np.copy()
    all_quads, all_lkps, all_rkps = [], [], []

    for _ in range(n_passes):
        if (current > 0.5).sum() < min_area:
            break
        quads, lkps, rkps = _process_curved(
            current, sample_pts, dp_epsilon, max_sub, min_gap,
            min_area, straight_threshold, border_offset_px)
        if not quads:
            break
        all_quads.extend(quads)
        all_lkps.extend(lkps)
        all_rkps.extend(rkps)

        # Retirer les pixels couverts du masque courant
        cov = _quads_to_coverage(quads, H, W)
        current = current.copy()
        current[cov > 0] = 0.0

    return all_quads, all_lkps, all_rkps


# ── Drawing helpers ────────────────────────────────────────────────────────────

def _draw_quad(canvas, quad, color, alpha, border_width, draw_fill=True,
               mask_clip=None, clip_borders=False, mask_dilated=None):
    """
    Dessine un quad sur canvas.
    mask_clip    : si fourni, le fill est restreint aux pixels du masque.
    clip_borders : si True, les bords sont aussi clippés au masque dilaté.
    """
    pts = quad.astype(np.int32)
    bright = tuple(min(255, int(c * 1.4)) for c in color)
    H, W = canvas.shape[:2]

    if draw_fill:
        if mask_clip is not None:
            quad_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(quad_mask, [pts], 255)
            fill_region = (quad_mask > 0) & (mask_clip > 0.5)
            overlay = canvas.copy()
            overlay[fill_region] = color
            alpha_map = np.where(fill_region, alpha, 0.0)[:, :, None]
            canvas = (canvas * (1 - alpha_map) + overlay * alpha_map).astype(np.uint8)
        else:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            canvas = cv2.addWeighted(canvas, 1 - alpha, overlay, alpha, 0)

    # Bords gauche/droit
    for p0, p1 in [(pts[0], pts[3]), (pts[1], pts[2])]:
        if clip_borders and mask_dilated is not None:
            # Dessiner la ligne sur un canvas temporaire puis masquer
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.line(tmp, tuple(p0), tuple(p1), 255, border_width, cv2.LINE_AA)
            tmp[mask_dilated == 0] = 0
            canvas[tmp > 0] = bright
        else:
            cv2.line(canvas, tuple(p0), tuple(p1), bright, border_width, cv2.LINE_AA)

    # Séparateurs haut/bas — fins, sans clipping
    cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(canvas, tuple(pts[3]), tuple(pts[2]), (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


# ── ComfyUI Node ───────────────────────────────────────────────────────────────

class RoadQuadSegments:
    """
    [TEST] Segmente chaque route en quads.

    Route DROITE  : régression linéaire globale + offset.
    Route COURBÉE : DP point-par-point. Segments localement droits → line fit.

    Deux images de debug :
      debug_image  : fill coloré sur tout le quad
      masked_debug : fill uniquement là où le masque confirme la route,
                     lignes de bord toujours visibles sur toute la longueur
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "individual_masks": ("MASK",),
                "straightness_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ratio s1/s0 PCA. En dessous = route droite (line fit global). "
                               "Aussi utilisé par segment pour les routes courbées."}),
                "dp_epsilon": ("FLOAT", {
                    "default": 25.0, "min": 1.0, "max": 300.0, "step": 1.0,
                    "tooltip": "Epsilon Douglas-Peucker pour les routes courbées."}),
                "max_segs_between_keypoints": ("INT", {
                    "default": 2, "min": 1, "max": 20}),
                "sample_points": ("INT", {
                    "default": 300, "min": 50, "max": 2000}),
                "min_keypoint_gap": ("INT", {
                    "default": 10, "min": 2, "max": 200}),
                "border_offset_px": ("INT", {
                    "default": 3, "min": -50, "max": 100,
                    "tooltip": "Décalage des bords estimés vers l'extérieur (pixels). "
                               "Compense le mask rétréci par rapport à la vraie route."}),
                "alpha": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_area_px": ("INT", {
                    "default": 200, "min": 10, "max": 100000}),
                "n_passes": ("INT", {
                    "default": 2, "min": 1, "max": 10,
                    "tooltip": "Passes hiérarchiques. Pass 1 trouve la composante principale, "
                               "pass 2 relance le même algo sur le masque résiduel (initial − déjà trouvé), "
                               "etc. Permet de détecter tous les virages d'une route."}),
            },
            "optional": {
                "background_image": ("IMAGE",),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "INT", ROAD_QUAD_PAIRS)
    RETURN_NAMES  = ("debug_image", "masked_debug", "n_total_quads", "quad_pairs")
    FUNCTION      = "segment"
    CATEGORY      = "road/insertion"

    def segment(self, individual_masks, straightness_threshold=0.05,
                dp_epsilon=25.0, max_segs_between_keypoints=2,
                sample_points=300, min_keypoint_gap=10, border_offset_px=3,
                alpha=0.35, min_area_px=200, n_passes=2, background_image=None):

        masks_t = individual_masks
        if masks_t.dim() == 2:
            masks_t = masks_t.unsqueeze(0)
        N, H, W = masks_t.shape

        masks_np = _resolve_overlap_pairwise(
            [masks_t[i].cpu().numpy() for i in range(N)])

        if background_image is not None:
            bg = (background_image[0].cpu().float().numpy() * 255
                  ).astype(np.uint8)[..., :3].copy()
        else:
            bg = np.zeros((H, W, 3), dtype=np.uint8)
            for m in masks_np:
                bg[m > 0.5] = 50

        canvas       = bg.copy()   # debug_image  : fill complet
        canvas_mask  = bg.copy()   # masked_debug : fill clipé au masque

        total_quads = 0
        quad_pairs  = []   # liste de dicts pour UniDepthRoadWidth

        for road_idx, mask_np in enumerate(masks_np):
            color = _ROAD_COLORS[road_idx % len(_ROAD_COLORS)]
            shade = tuple(int(c * 0.7) for c in color)

            ratio, _, _ = _pca_straightness(mask_np)
            is_straight  = ratio < straightness_threshold

            if is_straight:
                left_pts, right_pts = _border_points_all_fragments(mask_np, min_area_px)
                if left_pts is None:
                    continue
                left_r,  dl, _ = _fit_line_side(left_pts,  sample_points)
                right_r, dr, _ = _fit_line_side(right_pts, sample_points)
                if border_offset_px != 0:
                    perp = _perp_from_dir(dl, dr)
                    left_r  = _offset_side(left_r,  -perp, border_offset_px)
                    right_r = _offset_side(right_r,  perp, border_offset_px)
                quads    = _quads_from_sides(left_r, right_r, max_segs_between_keypoints)
                left_kps, right_kps = [], []
                mode = f"LINE FIT  r={ratio:.3f}"
                bw = 1
            else:
                quads, left_kps, right_kps = _process_curved_hierarchical(
                    mask_np, n_passes, sample_points, dp_epsilon,
                    max_segs_between_keypoints, min_keypoint_gap, min_area_px,
                    straightness_threshold, border_offset_px)
                mode = f"DP×{n_passes}  r={ratio:.3f}"
                bw = 1

            total_quads += len(quads)

            # Masque dilaté pour le clipping des bords (masked_debug)
            mask_u8 = (mask_np > 0.5).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask_dilated = cv2.dilate(mask_u8, kernel)

            for seg_idx, quad in enumerate(quads):
                quad_pairs.append({
                    "quad":     quad,          # (4,2) float32 : [left_top, right_top, right_bot, left_bot]
                    "road_idx": road_idx,
                    "seg_idx":  seg_idx,
                    "color":    color,
                })
                shade_i = tuple(max(0, int(c * (0.55 + seg_idx * 0.04))) for c in color)
                shade_i = tuple(min(255, v) for v in shade_i)

                canvas      = _draw_quad(canvas,      quad, shade_i, alpha, bw,
                                         draw_fill=True,  mask_clip=None,
                                         clip_borders=False)
                canvas_mask = _draw_quad(canvas_mask, quad, shade_i, alpha, bw,
                                         draw_fill=True,  mask_clip=mask_np,
                                         clip_borders=True, mask_dilated=mask_dilated)

                cx = int(quad[:, 0].mean())
                cy = int(quad[:, 1].mean())
                label = f"{'S' if is_straight else 'C'}{road_idx+1}-{seg_idx+1}"
                for c in (canvas, canvas_mask):
                    cv2.putText(c, label, (cx - 20, cy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,255,255), 1, cv2.LINE_AA)

            for kp in left_kps:
                for c in (canvas, canvas_mask):
                    cv2.circle(c, tuple(kp.tolist()), 5, (255, 230, 0), -1)
            for kp in right_kps:
                for c in (canvas, canvas_mask):
                    cv2.circle(c, tuple(kp.tolist()), 5, (255, 140, 0), -1)

            ys_m, xs_m = np.where(mask_np > 0.5)
            if len(xs_m) > 0:
                mx, my = int(xs_m.mean()), int(ys_m.mean())
                col_l = (255, 255, 80) if is_straight else (160, 200, 255)
                for c in (canvas, canvas_mask):
                    cv2.putText(c, mode, (mx - 40, my - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_l, 1, cv2.LINE_AA)

        info = f"{N} routes | {total_quads} quads | thr={straightness_threshold}"
        for c in (canvas, canvas_mask):
            cv2.putText(c, info, (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (180, 180, 180), 1, cv2.LINE_AA)

        def _to_t(arr):
            return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

        return (_to_t(canvas), _to_t(canvas_mask), total_quads, quad_pairs)


NODE_CLASS_MAPPINGS        = {"RoadQuadSegments": RoadQuadSegments}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadQuadSegments": "TEST"}
