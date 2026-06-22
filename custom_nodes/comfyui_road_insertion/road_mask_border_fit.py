"""
RoadMaskBorderFit
=================
1. Connected components → garde le PLUS GRAND blob (route principale)
2. Scan bord gauche / bord droit ligne par ligne
3. Régression linéaire sur des fenêtres glissantes → petits segments
4. Merge des segments proches et alignés
5. Garde les 2 plus longs (un gauche, un droit)
→ Sort LINE_SEGMENTS compatible avec Road Gravity Sampler
"""

import numpy as np
import cv2
import torch


# ─── helpers image ────────────────────────────────────────────────────────────

def _t2np(t):
    if t.dim() == 4:
        t = t[0]
    return (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)

def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

def _extract_mask(rgb, thr):
    """Accepte masque vert (SAM) ou masque blanc/noir binaire."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    green = (g > r + thr) & (g > b + thr) & (g > 60)
    white = (r > 127) & (g > 127) & (b > 127)
    return (green | white).astype(np.uint8)


# ─── 1. Plus grand connected component ───────────────────────────────────────

def _largest_cc(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return mask
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == best).astype(np.uint8)


# ─── 2. Scan bords gauche / droit par ligne ───────────────────────────────────

def _scan_borders(mask):
    rows, left_x, right_x = [], [], []
    for r in range(mask.shape[0]):
        cols = np.where(mask[r] > 0)[0]
        if len(cols) < 2:
            continue
        rows.append(float(r))
        left_x.append(float(cols[0]))
        right_x.append(float(cols[-1]))
    return (np.array(rows), np.array(left_x), np.array(right_x))


# ─── 3. Petits segments par fenêtre glissante ─────────────────────────────────

def _windowed_segments(rows, xs, window, step):
    """
    Régression linéaire sur des fenêtres de `window` lignes, avançant de `step`.
    Retourne liste de np.array([[x0,y0],[x1,y1]]).
    """
    segs = []
    n = len(rows)
    i = 0
    while i + window <= n:
        r_win = rows[i:i + window]
        x_win = xs[i:i + window]
        try:
            a, b = np.polyfit(r_win, x_win, 1)
        except Exception:
            i += step
            continue
        y0, y1 = float(r_win[0]), float(r_win[-1])
        segs.append(np.array([[a * y0 + b, y0],
                               [a * y1 + b, y1]], np.float32))
        i += step
    return segs


# ─── 4. Merge des segments proches et alignés ─────────────────────────────────

def _seg_angle(s):
    dx = s[1, 0] - s[0, 0]
    dy = s[1, 1] - s[0, 1] + 1e-9
    return float(np.degrees(np.arctan2(dx, dy))) % 180.0

def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)

def _merge_segments(segs, angle_tol, gap_tol):
    """
    Fusionne les segments consécutifs dont l'angle diffère de moins de angle_tol
    et dont le gap vertical est inférieur à gap_tol.
    """
    if not segs:
        return []
    # Trier par y de départ
    segs = sorted(segs, key=lambda s: s[0, 1])
    merged = [segs[0].copy()]
    for s in segs[1:]:
        prev = merged[-1]
        gap = float(s[0, 1] - prev[1, 1])
        if gap <= gap_tol and _angle_diff(_seg_angle(prev), _seg_angle(s)) <= angle_tol:
            # Prolonger le segment précédent jusqu'à la fin de s
            # Refaire une régression sur l'union des deux... simplement on étend
            merged[-1][1] = s[1].copy()
        else:
            merged.append(s.copy())
    return merged


# ─── 5. Longueur d'un segment ────────────────────────────────────────────────

def _seg_len(s):
    return float(np.hypot(s[1, 0] - s[0, 0], s[1, 1] - s[0, 1]))


# ─── 6. Prolonger un segment jusqu'à un y-range cible ───────────────────────

def _extend_segment(seg, target_y_min, target_y_max):
    """
    Prolonge `seg` géométriquement pour couvrir [target_y_min, target_y_max] en Y.
    Suit simplement la pente du segment, sans vérification de masque.
    """
    s = seg.copy().astype(np.float32)

    # S'assurer que s[0] est le point le plus haut (y le plus petit)
    if s[0, 1] > s[1, 1]:
        s = s[::-1].copy()

    dy = s[1, 1] - s[0, 1]
    dx = s[1, 0] - s[0, 0]
    if abs(dy) < 1e-6:
        return s  # segment horizontal, pas de prolongation

    slope = dx / dy  # dX par dY

    # Prolonger vers le haut
    if s[0, 1] > target_y_min:
        delta = s[0, 1] - target_y_min
        s[0] = np.array([s[0, 0] - slope * delta, target_y_min], np.float32)

    # Prolonger vers le bas
    if s[1, 1] < target_y_max:
        delta = target_y_max - s[1, 1]
        s[1] = np.array([s[1, 0] + slope * delta, target_y_max], np.float32)

    return s


# ─── debug ────────────────────────────────────────────────────────────────────

def _draw_debug(rgb, road_mask, all_l, all_r, best_l, best_r):
    vis = rgb.copy()
    # Colorier la route retenue en vert
    vis[road_mask > 0] = np.clip(
        vis[road_mask > 0].astype(np.float32) * 0.35
        + np.array([20, 180, 50]) * 0.65, 0, 255).astype(np.uint8)

    # Tous les petits segments : gris
    for s in all_l + all_r:
        cv2.line(vis, tuple(np.round(s[0]).astype(int)),
                 tuple(np.round(s[1]).astype(int)), (80, 80, 80), 1, cv2.LINE_AA)

    # Meilleurs segments
    def draw_seg(seg, color):
        if seg is None:
            return
        p0 = tuple(np.round(seg[0]).astype(int))
        p1 = tuple(np.round(seg[1]).astype(int))
        cv2.line(vis, p0, p1, color, 3, cv2.LINE_AA)
        cv2.circle(vis, p0, 6, color, -1, cv2.LINE_AA)
        cv2.circle(vis, p1, 6, color, -1, cv2.LINE_AA)

    draw_seg(best_l, (255, 140, 0))   # orange = gauche
    draw_seg(best_r, (0, 160, 255))   # bleu   = droite

    cv2.putText(vis, "left border",  (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 140, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, "right border", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 160, 255), 2, cv2.LINE_AA)
    return vis


# ─── node ComfyUI ─────────────────────────────────────────────────────────────

class RoadMaskBorderFit:
    CATEGORY = "road/insertion"
    FUNCTION = "fit"
    RETURN_TYPES = ("IMAGE", "LINE_SEGMENTS", "STRING")
    RETURN_NAMES = ("debug_image", "border_lines", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),
            },
            "optional": {
                "green_threshold": ("INT", {
                    "default": 80, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Seuil pour masque vert. Ignoré pour masques B&W."}),
                "window_rows": ("INT", {
                    "default": 40, "min": 5, "max": 300, "step": 5,
                    "tooltip": "Taille de la fenêtre glissante (lignes) pour chaque mini-régression."}),
                "step_rows": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Pas d'avancement de la fenêtre glissante."}),
                "merge_angle_deg": ("FLOAT", {
                    "default": 8.0, "min": 0.5, "max": 45.0, "step": 0.5,
                    "tooltip": "Angle max (°) entre deux segments consécutifs pour les fusionner."}),
                "merge_gap_rows": ("INT", {
                    "default": 15, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Gap vertical max (lignes) entre deux segments pour les fusionner."}),
                "min_road_pixels": ("INT", {
                    "default": 100, "min": 10, "max": 100000, "step": 10}),
                "extend_to_equal": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Prolonge le segment le plus court pour qu'il atteigne "
                               "le même y-range que le plus long, en restant dans le masque route."}),
            },
        }

    def fit(self, mask_image,
            green_threshold=80, window_rows=40, step_rows=10,
            merge_angle_deg=8.0, merge_gap_rows=15, min_road_pixels=100,
            extend_to_equal=False):

        rgb  = _t2np(mask_image)
        H, W = rgb.shape[:2]
        empty_lines = {"lines": np.zeros((0, 2, 2), np.float32), "width": W, "height": H}

        # ── 1. Masque + plus grand composant connexe ──────────────────────────
        mask = _extract_mask(rgb, green_threshold)
        mask = _largest_cc(mask)

        n_px = int(mask.sum())
        if n_px < min_road_pixels:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(vis, f"Trop peu de pixels ({n_px})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return (_np2t(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)),
                    empty_lines, f"Trop peu de pixels ({n_px})")

        # ── 2. Scan bords ──────────────────────────────────────────────────────
        rows, left_x, right_x = _scan_borders(mask)
        if len(rows) < window_rows:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(vis, "Trop peu de lignes valides", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            return (_np2t(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)),
                    empty_lines, "Trop peu de lignes valides")

        # ── 3. Petits segments glissants ───────────────────────────────────────
        raw_l = _windowed_segments(rows, left_x,  window_rows, step_rows)
        raw_r = _windowed_segments(rows, right_x, window_rows, step_rows)

        # ── 4. Merge ───────────────────────────────────────────────────────────
        merged_l = _merge_segments(raw_l, merge_angle_deg, merge_gap_rows)
        merged_r = _merge_segments(raw_r, merge_angle_deg, merge_gap_rows)

        # ── 5. Garder le plus long de chaque côté ─────────────────────────────
        best_l = max(merged_l, key=_seg_len) if merged_l else None
        best_r = max(merged_r, key=_seg_len) if merged_r else None

        # ── 6. Extension égale (optionnel) ────────────────────────────────────
        if extend_to_equal and best_l is not None and best_r is not None:
            # y-range union des deux segments
            yl0 = min(best_l[0, 1], best_l[1, 1])
            yl1 = max(best_l[0, 1], best_l[1, 1])
            yr0 = min(best_r[0, 1], best_r[1, 1])
            yr1 = max(best_r[0, 1], best_r[1, 1])
            target_y0 = min(yl0, yr0)
            target_y1 = max(yl1, yr1)
            best_l = _extend_segment(best_l, target_y0, target_y1)
            best_r = _extend_segment(best_r, target_y0, target_y1)

        kept = [s for s in [best_l, best_r] if s is not None]
        lines_arr = np.stack(kept, axis=0) if kept else np.zeros((0, 2, 2), np.float32)
        border_lines = {"lines": lines_arr, "width": W, "height": H}

        # ── 7. Debug ───────────────────────────────────────────────────────────
        vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        vis_bgr = _draw_debug(vis_bgr, mask, merged_l, merged_r, best_l, best_r)

        summary = (f"RoadMaskBorderFit | {n_px} px | "
                   f"L: {len(merged_l)} segs → best {_seg_len(best_l):.0f}px | "
                   f"R: {len(merged_r)} segs → best {_seg_len(best_r):.0f}px"
                   if best_l is not None and best_r is not None
                   else f"{n_px} px | L:{len(merged_l)} R:{len(merged_r)}")

        cv2.putText(vis_bgr, summary[:90], (8, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        return (_np2t(cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)), border_lines, summary)


NODE_CLASS_MAPPINGS        = {"RoadMaskBorderFit": RoadMaskBorderFit}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadMaskBorderFit": "Road Mask Border Fit"}
