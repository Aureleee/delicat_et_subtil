"""
ComfyUI Custom Node : Local Border Reconstruction & Line Fitting  (Node 3 / 3)
==============================================================================
From the contours validated by Node 2, rebuild geometrically coherent road
borders as straight line segments.

The approach is strictly LOCAL -- never a global regression over the whole
contour -- so it survives bends, right angles, intersections and several roads
sharing the image:

  1. cut the valid contour into runs of consecutive RELIABLE points;
  2. each run already walks ONE side of a portion of carriageway, so the two
     opposite sides are separated naturally by the cuts/corners;
  3. simplify each run (Douglas-Peucker) -> short straight pieces, with a
     vertex created at every real direction break (turns / 90 deg corners);
  4. fit / merge pieces that are collinear AND close into longer segments;
  5. keep the significant direction breaks (collinear merge never crosses a
     real corner).

Output
------
  - debug_image  : faint contours + the fitted border segments (coloured)
  - border_lines : LINE_SEGMENTS  -> plugs straight into the rest of the pack
  - summary      : STRING
"""

import numpy as np
import cv2

from .line_helpers import (
    numpy_to_comfy_image, make_line_segments, empty_line_segments, mask_to_bool,
)


def _seg_angle(a, b):
    """Angle of segment a->b in [0,180)."""
    ang = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0
    return ang


def _angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _fit_segment(pts):
    """PCA line fit through pts -> (segment [[x0,y0],[x1,y1]], max_perp_dist)."""
    pts = np.asarray(pts, np.float32)
    c = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0]                                   # principal direction
    t = (pts - c) @ d
    p0 = c + d * t.min()
    p1 = c + d * t.max()
    # perpendicular spread
    nvec = np.array([-d[1], d[0]], np.float32)
    perp = np.abs((pts - c) @ nvec)
    return np.array([p0, p1], np.float32), float(perp.max() if perp.size else 0.0)


def _arc_reliable_frac(rel, i0, i1):
    """Mean reliability of the contour arc from index i0 to i1 (wrapping)."""
    n = rel.size
    if n == 0:
        return 1.0
    if i1 >= i0:
        seg = rel[i0:i1 + 1]
    else:
        seg = np.concatenate([rel[i0:], rel[:i1 + 1]])
    return float(seg.mean()) if seg.size else 1.0


def _merge_collinear(segs, angle_tol, gap_tol, perp_tol):
    """Greedily merge near-collinear, near-touching segments."""
    segs = [np.asarray(s, np.float32) for s in segs]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        n = len(segs)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = segs[i], segs[j]
                ai = _seg_angle(a[0], a[1])
                bj = _seg_angle(b[0], b[1])
                if _angle_diff(ai, bj) > angle_tol:
                    continue
                pts = np.vstack([a, b])
                cand, perp = _fit_segment(pts)
                if perp > perp_tol:
                    continue
                # the four endpoints must project to a single contiguous span:
                # i.e. the two segments must overlap or be within gap_tol.
                d = cand[1] - cand[0]
                L = np.hypot(*d)
                if L < 1e-6:
                    continue
                u = d / L
                ta = sorted([(a[0] - cand[0]) @ u, (a[1] - cand[0]) @ u])
                tb = sorted([(b[0] - cand[0]) @ u, (b[1] - cand[0]) @ u])
                gap = max(tb[0] - ta[1], ta[0] - tb[1])     # >0 only if disjoint
                if gap > gap_tol:
                    continue
                segs[i] = cand
                del segs[j]
                changed = True
                break
            if changed:
                break
    return segs


def _pair_score(sa, sb, max_lane_width):
    """
    Score une paire (sa, sb) supposée être les deux bordures d'une même route.

    Critères (tous doivent être cohérents avec "même route") :
    1. Longueur : on favorise les deux segments longs → score ∝ min(len_a, len_b)
    2. Chevauchement longitudinal : les deux bordures doivent se "couvrir" sur l'axe
       de la route → score × overlap_fraction (0..1)
    3. Écart transversal : doit être plausible pour une route (pas trop large, pas 0)
       → pénalité douce si > max_lane_width

    Retourne un float ≥ 0 (plus grand = meilleure paire).
    """
    sa = np.asarray(sa, np.float32)
    sb = np.asarray(sb, np.float32)
    la = np.hypot(*(sa[1] - sa[0]))
    lb = np.hypot(*(sb[1] - sb[0]))
    if la < 1e-6 or lb < 1e-6:
        return 0.0

    # Axe longitudinal commun = direction moyenne des deux segments
    da = (sa[1] - sa[0]) / la
    db = (sb[1] - sb[0]) / lb
    if np.dot(da, db) < 0:          # aligner dans le même sens
        db = -db
    d_avg = da + db
    d_norm = np.linalg.norm(d_avg)
    if d_norm < 1e-6:
        return 0.0
    d_avg /= d_norm
    perp = np.array([-d_avg[1], d_avg[0]], np.float32)

    # Écart transversal entre les deux milieux
    mid_a = (sa[0] + sa[1]) * 0.5
    mid_b = (sb[0] + sb[1]) * 0.5
    transversal_gap = abs(float((mid_b - mid_a) @ perp))
    if transversal_gap < 1.0:       # les deux segments se superposent → pas des bordures opposées
        return 0.0

    # Projections longitudinales des extrémités sur d_avg
    def proj_range(s):
        ts = [(s[0] - mid_a) @ d_avg, (s[1] - mid_a) @ d_avg]
        return min(ts), max(ts)

    t_a0, t_a1 = proj_range(sa)
    t_b0, t_b1 = proj_range(sb)
    overlap = max(0.0, min(t_a1, t_b1) - max(t_a0, t_b0))
    union   = max(t_a1, t_b1) - min(t_a0, t_b0)
    overlap_frac = overlap / union if union > 1e-6 else 0.0

    # Pénalité si l'écart transversal est anormalement grand
    width_penalty = 1.0 / (1.0 + max(0.0, transversal_gap - max_lane_width) / max_lane_width)

    return min(la, lb) * overlap_frac * width_penalty


def _pick_two_sides(segs, max_lane_width=600.0):
    """
    Sélectionne la meilleure PAIRE de segments (un par côté) bordant la même route.

    Stratégie :
    1. PCA globale pour l'axe longitudinal → séparer gauche / droite.
    2. Pour chaque paire (un gauche, un droit), calculer _pair_score().
    3. Garder la paire qui maximise le score.
       → favorise naturellement les segments longs, qui se chevauchent bien
         en longueur et dont l'écart transversal est cohérent avec une route.

    Retourne (kept, left_idx, right_idx).
    """
    if len(segs) <= 1:
        return list(segs), (0 if segs else None), None

    mids = np.array([(s[0] + s[1]) * 0.5 for s in segs], np.float32)
    lens = np.array([np.hypot(*(s[1] - s[0])) for s in segs], np.float32)

    # Axe longitudinal global (PCA sur tous les endpoints)
    pts = np.vstack([np.asarray(s, np.float32) for s in segs])
    c   = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    d   = vt[0]
    nrm = np.array([-d[1], d[0]], np.float32)

    side = (mids - c) @ nrm
    neg_idx = np.where(side < 0)[0]   # côté "gauche"
    pos_idx = np.where(side >= 0)[0]  # côté "droit"

    # Cas dégénérés : tous les segments d'un seul côté
    if neg_idx.size == 0 or pos_idx.size == 0:
        all_idx = np.arange(len(segs))
        best = int(np.argmax(lens))
        return [segs[best]], 0, None

    # Tester toutes les paires (gauche × droite) et garder la meilleure
    best_score = -1.0
    best_left  = int(neg_idx[np.argmax(lens[neg_idx])])   # fallback : le plus long
    best_right = int(pos_idx[np.argmax(lens[pos_idx])])

    for i in neg_idx:
        for j in pos_idx:
            sc = _pair_score(segs[i], segs[j], max_lane_width)
            if sc > best_score:
                best_score = sc
                best_left, best_right = i, j

    kept = [segs[best_left], segs[best_right]]
    # gauche = milieu X le plus petit
    if (segs[best_left][0][0] + segs[best_left][1][0]) > \
       (segs[best_right][0][0] + segs[best_right][1][0]):
        kept = [segs[best_right], segs[best_left]]
    return kept, 0, 1


class LocalBorderReconstruction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clean_contours": ("ROAD_CONTOURS",),
            },
            "optional": {
                "road_mask": ("IMAGE",),
                "simplify_eps_px": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 50.0, "step": 0.5}),
                "min_reliable_frac": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_segment_px": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 2000.0, "step": 1.0}),
                "merge_angle_deg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 45.0, "step": 0.5}),
                "merge_gap_px": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 1000.0, "step": 1.0}),
                "merge_perp_px": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "max_two_sides": ("BOOLEAN", {"default": False}),
                "max_lane_width_px": ("FLOAT", {
                    "default": 600.0, "min": 50.0, "max": 3000.0, "step": 10.0,
                    "tooltip": "Largeur max acceptable entre les deux bordures (pixels). "
                               "Evite de sélectionner deux segments de routes différentes."}),
                "mask_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = ("IMAGE", "LINE_SEGMENTS", "STRING")
    RETURN_NAMES = ("debug_image", "border_lines", "summary")
    FUNCTION = "reconstruct"
    CATEGORY = "DeepLSD Road/3 Contours"

    def reconstruct(self, clean_contours, road_mask=None, simplify_eps_px=4.0,
                    min_reliable_frac=0.6, min_segment_px=50.0, merge_angle_deg=5.0,
                    merge_gap_px=20.0, merge_perp_px=20.0, max_two_sides=False,
                    max_lane_width_px=600.0, mask_threshold=10):
        w = int(clean_contours.get("width", 0))
        h = int(clean_contours.get("height", 0))
        cont_list = clean_contours.get("contours", [])
        hole_list = clean_contours.get("is_hole", [False] * len(cont_list))
        rel_list = clean_contours.get("reliable", None)

        raw_segs = []
        for idx, pts in enumerate(cont_list):
            if hole_list[idx]:
                continue
            n = pts.shape[0]
            if n < 2:
                continue
            if rel_list is not None and rel_list[idx] is not None:
                rel = np.asarray(rel_list[idx], bool)
            else:
                rel = np.ones(n, bool)            # Node 2 skipped -> trust all

            # 1) simplify the FULL contour: corners become vertices
            approx = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2),
                                      float(simplify_eps_px), True).reshape(-1, 2)
            m = len(approx)
            if m < 2:
                continue
            # map each vertex back to its index on the original contour
            vidx = [int(np.argmin(np.hypot(pts[:, 0] - v[0], pts[:, 1] - v[1])))
                    for v in approx]
            # 2) keep each edge only if its underlying arc is mostly reliable
            for k in range(m):
                a, b = approx[k], approx[(k + 1) % m]
                if np.hypot(*(b - a)) < min_segment_px:
                    continue
                if _arc_reliable_frac(rel, vidx[k], vidx[(k + 1) % m]) < min_reliable_frac:
                    continue
                raw_segs.append(np.array([a, b], np.float32))

        # 3) merge collinear, close pieces (real corners are never merged)
        segs = _merge_collinear(raw_segs, merge_angle_deg, merge_gap_px, merge_perp_px)
        segs = [s for s in segs if np.hypot(*(s[1] - s[0])) >= min_segment_px]

        # 4) optional: keep at most two sides (longest on each side of the axis)
        left_idx = right_idx = None
        if max_two_sides:
            segs, left_idx, right_idx = _pick_two_sides(segs, max_lane_width=float(max_lane_width_px))

        # --- debug ---
        debug = np.zeros((h, w, 3), np.uint8)
        if road_mask is not None:
            mb = mask_to_bool(road_mask, 0, mask_threshold)
            if mb.shape == (h, w):
                debug[mb] = (28, 45, 28)
        for pts in cont_list:                      # faint reference contours
            cv2.polylines(debug, [np.round(pts).astype(np.int32)], True,
                          (70, 70, 70), 1, cv2.LINE_AA)
        palette = [(0, 220, 70), (255, 150, 0), (0, 160, 255),
                   (255, 80, 200), (240, 240, 0), (0, 255, 200)]
        for i, s in enumerate(segs):
            if max_two_sides:                      # left = orange, right = blue
                col = (255, 150, 0) if i == left_idx else (0, 160, 255)
            else:
                col = palette[i % len(palette)]
            p0 = tuple(np.round(s[0]).astype(int))
            p1 = tuple(np.round(s[1]).astype(int))
            cv2.line(debug, p0, p1, col, 3, cv2.LINE_AA)
            cv2.circle(debug, p0, 4, col, -1)
            cv2.circle(debug, p1, 4, col, -1)

        if segs:
            lines = make_line_segments(np.stack(segs, axis=0), w, h)
        else:
            lines = empty_line_segments(w, h)

        if max_two_sides:
            sides = ("left" if left_idx is not None else "-") + "/" + \
                    ("right" if right_idx is not None else "-")
            summary = (f"Reconstruction: {len(raw_segs)} local piece(s) -> "
                       f"two-sides mode -> {len(segs)} border(s) kept ({sides}).")
        else:
            summary = (f"Reconstruction: {len(raw_segs)} local piece(s) -> "
                       f"{len(segs)} merged border segment(s).")
        return (numpy_to_comfy_image(debug), lines, summary)
