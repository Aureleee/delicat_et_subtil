"""
ComfyUI Custom Node : Corrupted Border Detection  (Node 2 / 3)
==============================================================
From the contours of Node 1, decide which portions are trustworthy and which
are corrupted by holes in the mask.

Idea
----
Estimate the local road width by linking the two opposite borders with a
transversal chord crossing the carriageway. For each contour point:
  - shoot a chord INWARD (perpendicular to the local contour tangent);
  - read the mask along that chord.
A clean chord crosses ONE solid run of road, then exits at the opposite
border. If instead the chord meets road -> empty -> road again (a 1-0-1
pattern) inside the expected road width, a hole is crossing it: that contour
portion is marked SUSPECT.

Hole-boundary contours (inner contours from Node 1) are intrinsically not real
road borders and are marked corrupted as well.

Outputs
-------
  - debug_image       : reliable contour (green), corrupted (red),
                        detected holes (yellow dots + faint chords)
  - clean_contours    : ROAD_CONTOURS  with a per-point `reliable` mask
                        (consumed by "Local Border Reconstruction")
  - summary           : STRING
"""

import numpy as np
import cv2

from .line_helpers import numpy_to_comfy_image, mask_to_bool


def _tangents(pts, k):
    """Local unit tangent at each point of a (closed) polyline, shape (N,2)."""
    n = pts.shape[0]
    if n < 2:
        return np.zeros_like(pts)
    fwd = np.roll(pts, -k, axis=0)
    bwd = np.roll(pts, k, axis=0)
    t = fwd - bwd
    norm = np.hypot(t[:, 0], t[:, 1])
    norm[norm < 1e-6] = 1.0
    return t / norm[:, None]


def _ray_inside(mask_bool, P, n, start, maxd):
    """Sample mask along P + n*t for t in [start, maxd) at 1px steps -> bool[]."""
    h, w = mask_bool.shape
    ts = np.arange(start, maxd, 1.0, dtype=np.float32)
    xs = np.round(P[0] + n[0] * ts).astype(np.int32)
    ys = np.round(P[1] + n[1] * ts).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    vals = np.zeros(ts.shape[0], dtype=bool)
    vals[valid] = mask_bool[ys[valid], xs[valid]]
    return ts, vals


def _leading_run(vals):
    """Length of the leading True run."""
    if vals.size == 0 or not vals[0]:
        return 0
    idx = np.argmax(~vals)            # first False
    return int(idx) if vals[idx] == False else int(vals.size)


def _classify_chord(vals, max_gap_px, reentry_min):
    """
    Return (reliable: bool, hole_t: int | None).
    A hole => road run, then a bounded empty gap, then road again.
    """
    n = vals.size
    if n == 0 or not vals[0]:
        return False, None                      # could not enter the road
    i = 0
    while i < n and vals[i]:                     # leading road run
        i += 1
    if i >= n:
        return True, None                        # road all the way (thin chord) -> ok
    gap_start = i
    while i < n and not vals[i]:                 # empty run
        i += 1
    gap_len = i - gap_start
    if i >= n:
        return True, None                        # reached the opposite border -> reliable
    # there is road again after the gap
    reentry = 0
    while i < n and vals[i]:
        reentry += 1
        i += 1
    if gap_len <= max_gap_px and reentry >= reentry_min:
        return False, int((gap_start + gap_start + gap_len) // 2)   # HOLE
    return True, None                            # gap too wide -> separate region, border ok


class CorruptedBorderDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "contours": ("ROAD_CONTOURS",),
                "road_mask": ("IMAGE",),
            },
            "optional": {
                "step_px": ("INT", {"default": 4, "min": 1, "max": 50}),
                "tangent_k": ("INT", {"default": 4, "min": 1, "max": 50}),
                "probe_px": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 20.0, "step": 0.5}),
                "max_width_px": ("FLOAT", {"default": 400.0, "min": 10.0, "max": 4000.0, "step": 10.0}),
                "max_hole_gap_px": ("FLOAT", {"default": 200.0, "min": 2.0, "max": 4000.0, "step": 2.0}),
                "reentry_min_px": ("INT", {"default": 3, "min": 1, "max": 200}),
                "spread": ("INT", {"default": 1, "min": 0, "max": 50}),
                "mask_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
            },
        }

    RETURN_TYPES = ("IMAGE", "ROAD_CONTOURS", "STRING")
    RETURN_NAMES = ("debug_image", "clean_contours", "summary")
    FUNCTION = "detect"
    CATEGORY = "DeepLSD Road/3 Contours"

    def detect(self, contours, road_mask, step_px=4, tangent_k=4, probe_px=2.0,
               max_width_px=400.0, max_hole_gap_px=200.0, reentry_min_px=3,
               spread=1, mask_threshold=10):
        mask_bool = mask_to_bool(road_mask, 0, mask_threshold)
        h, w = mask_bool.shape

        cont_list = contours.get("contours", [])
        hole_list = contours.get("is_hole", [None] * len(cont_list))

        out_contours, out_holes, out_reliable = [], [], []
        debug = np.zeros((h, w, 3), np.uint8)
        debug[mask_bool] = (30, 50, 30)
        holes_found = []
        n_pts_ok = n_pts_bad = 0

        for pts, is_hole in zip(cont_list, hole_list):
            n = pts.shape[0]
            reliable = np.zeros(n, dtype=bool)

            if is_hole:
                # hole boundary is never a real road border
                out_contours.append(pts)
                out_holes.append(True)
                out_reliable.append(reliable)         # all False
                n_pts_bad += n
                ip = np.round(pts).astype(np.int32)
                cv2.polylines(debug, [ip], True, (0, 0, 230), 2, cv2.LINE_AA)
                continue

            tans = _tangents(pts, tangent_k)
            for i in range(0, n, step_px):
                P = pts[i]
                tx, ty = tans[i]
                normals = (np.array([-ty, tx], np.float32),
                           np.array([ty, -tx], np.float32))
                # pick the inward normal (longer leading road run)
                best_n, best_lead, best_vals = None, -1, None
                for nrm in normals:
                    ts, vals = _ray_inside(mask_bool, P, nrm, probe_px, max_width_px)
                    lead = _leading_run(vals)
                    if lead > best_lead:
                        best_lead, best_n, best_vals = lead, nrm, vals
                if best_lead <= 0:
                    ok, hole_t = False, None
                else:
                    ok, hole_t = _classify_chord(best_vals, max_hole_gap_px, reentry_min_px)

                lo = max(0, i - spread)
                hi = min(n, i + spread + 1)
                reliable[lo:hi] = ok
                if not ok and hole_t is not None and best_n is not None:
                    hp = P + best_n * (probe_px + hole_t)
                    holes_found.append(hp)

            out_contours.append(pts)
            out_holes.append(False)
            out_reliable.append(reliable)
            n_pts_ok += int(reliable.sum())
            n_pts_bad += int((~reliable).sum())

            # draw per-point reliability
            ip = np.round(pts).astype(np.int32)
            for j in range(n - 1):
                col = (0, 220, 70) if reliable[j] else (0, 0, 230)
                cv2.line(debug, tuple(ip[j]), tuple(ip[j + 1]), col, 2, cv2.LINE_AA)

        for hp in holes_found:
            x, y = int(round(hp[0])), int(round(hp[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(debug, (x, y), 4, (0, 220, 255), -1, cv2.LINE_AA)

        out = {
            "contours": out_contours,
            "is_hole": out_holes,
            "reliable": out_reliable,
            "width": int(w),
            "height": int(h),
        }
        summary = (f"Borders: {n_pts_ok} reliable pts, {n_pts_bad} corrupted pts, "
                   f"{len(holes_found)} hole crossing(s) detected over "
                   f"{len(out_contours)} contour(s).")
        return (numpy_to_comfy_image(debug), out, summary)
