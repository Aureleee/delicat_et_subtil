"""
Road Orientation Estimator V2
==============================

STEP 1 — NORMAL
  SVD on 3D road points (x_norm, y_norm, depth).
  Depth is used HERE to reconstruct the 3D road plane.

STEP 2 — FORWARD (tangent)
  Centreline per-row (or per-column for flat roads) → PCA → 2D tangent.
  The forward 3D vector starts as (tang2d_x, tang2d_y, 0).
  Z=0 is intentional: the depth already contributed to the normal.
  Adding dz_ds here would make Z dominate (scale factor ≈ 0.003 vs dz_ds ≈ 0.1–0.5)
  and completely destroy the 2D direction.
  The correct Z component comes automatically from projecting onto the road plane.

STEP 3 — SIDE
  side = cross(normal, forward) in TRUE 3D.

Dependencies: torch, numpy, cv2
"""

import torch
import numpy as np
import cv2


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def t2np(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    return (np.clip(tensor.detach().cpu().float().numpy(), 0, 1) * 255).astype(np.uint8)

def np2t(arr):
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)

def fail_img(H, W, msg):
    img = np.zeros((H, W, 3), np.uint8); img[:, :, 2] = 60
    cv2.putText(img, "Road Orientation: FAILED", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
    words = msg.split(); lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 62: lines.append(cur); cur = w
        else: cur = (cur + " " + w).strip()
    if cur: lines.append(cur)
    for i, l in enumerate(lines):
        cv2.putText(img, l, (10, 62 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img

def draw_arrow(img, cx, cy, vec, color, label, scale, thick=2):
    ex, ey = int(cx + vec[0] * scale), int(cy + vec[1] * scale)
    cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey), color, thick,
                    tipLength=0.25, line_type=cv2.LINE_AA)
    cv2.putText(img, label, (ex + 5, ey - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Mask
# ═══════════════════════════════════════════════════════════════════════════════

def extract_green_mask(rgb, thr=80):
    r = rgb[:,:,0].astype(np.int16)
    g = rgb[:,:,1].astype(np.int16)
    b = rgb[:,:,2].astype(np.int16)
    return ((g > r + thr) & (g > b + thr) & (g > 80)).astype(np.uint8) * 255

def keep_largest_cc(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2: return mask
    best = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return (labels == best).astype(np.uint8) * 255


# ═══════════════════════════════════════════════════════════════════════════════
# Depth
# ═══════════════════════════════════════════════════════════════════════════════

def load_depth(img):
    d = img[:,:,:3].mean(2) if img.ndim == 3 else img
    return d.astype(np.float32) / 255.0

def smooth_depth(d, k):
    if k < 3: return d
    k = k if k % 2 else k + 1
    return cv2.GaussianBlur(d, (k, k), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Normal via SVD
# ═══════════════════════════════════════════════════════════════════════════════

def build_pts3d(mask, depth, step=4):
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if step > 1:
        i = np.arange(0, len(xs), step); xs, ys = xs[i], ys[i]
    if not len(xs): return np.empty((0, 3), np.float32)
    return np.stack([(xs / W) * 2.0 - 1.0,
                     (ys / H) * 2.0 - 1.0,
                     depth[ys, xs]], axis=1).astype(np.float32)

def svd_normal(pts):
    if len(pts) < 3: return None
    c = pts.mean(0)
    try: _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    except: return None
    n = vh[-1] / (np.linalg.norm(vh[-1]) + 1e-9)
    if n[1] > 0: n = -n   # ensure points upward (−Y in image-down coords)
    return n.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Forward via centreline PCA  (Z = 0, then project onto plane)
# ═══════════════════════════════════════════════════════════════════════════════

def _per_row_centreline(mask, H):
    cc, rr, lc, rc = [], [], [], []
    for row in range(H):
        px = np.where(mask[row] > 0)[0]
        if len(px) < 2: continue
        l, r = float(px.min()), float(px.max())
        lc.append(l); rc.append(r); cc.append((l+r)*0.5); rr.append(float(row))
    if not cc: return None, None, None, None, None
    cc = np.array(cc, np.float32); rr = np.array(rr, np.float32)
    return cc, rr, np.array(rc,np.float32)-np.array(lc,np.float32), np.array(lc,np.float32), np.array(rc,np.float32)

def _per_col_centreline(mask, W):
    cc, rr = [], []
    for col in range(W):
        px = np.where(mask[:, col] > 0)[0]
        if len(px) < 2: continue
        rr.append((float(px.min())+float(px.max()))*0.5); cc.append(float(col))
    if not cc: return None, None, None, None, None
    return np.array(cc,np.float32), np.array(rr,np.float32), None, None, None

def build_centreline(mask, H, W):
    ys, _ = np.where(mask > 0)
    if not len(ys): return None, None, None, None, None, False
    use_rows = (float(ys.max() - ys.min()) >= 0.2 * H)
    if use_rows:
        cc, rr, w, lc, rc = _per_row_centreline(mask, H)
    else:
        cc, rr, w, lc, rc = _per_col_centreline(mask, W)
    return cc, rr, w, lc, rc, use_rows

def centreline_pca_forward(mask, H, W):
    """
    1. Build per-row (or per-col) centreline.
    2. PCA on centreline → 2D unit tangent.
    3. Orient toward upper rows (far end in perspective).
    4. Return (tang2d_x, tang2d_y, 0) as the raw 3D forward before plane projection.
       Z=0 is correct here: depth already encoded in the normal.
       Adding dz_ds would make Z dominate the tiny scaled XY components (~0.003)
       and destroy the 2D road direction.
    """
    cc, rr, widths, lc, rc, used_rows = build_centreline(mask, H, W)
    if cc is None: return None, None, None, None, None

    cpts    = np.stack([cc, rr], axis=1).astype(np.float32)
    mean_px = cpts.mean(0)
    centered = cpts - mean_px

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None, None, None, None, None

    tang2d = vt[0].astype(np.float32)   # unit vector (Δcol, Δrow)

    # Orient: forward toward upper rows (smaller row = far end in perspective)
    if used_rows:
        arc   = centered @ tang2d
        ahead = arc > 0
        if ahead.sum() > 0 and (~ahead).sum() > 0:
            if rr[ahead].mean() > rr[~ahead].mean():
                tang2d = -tang2d
    # For per-column (nearly horizontal roads): keep as-is; orientation less critical

    # ── 3D forward: purely from 2D tangent, Z = 0 ────────────────────────────
    # tang2d is a unit vector in pixel space.
    # We treat it directly as the XY components of the 3D forward.
    # The correct Z component will emerge from projecting onto the road plane.
    fwd = np.array([tang2d[0], tang2d[1], 0.0], np.float32)

    return fwd, tang2d, mean_px, cc, rr


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Side via 3D cross product
# ═══════════════════════════════════════════════════════════════════════════════

def cross_side(normal, forward):
    s = np.cross(normal, forward).astype(np.float32)
    n = np.linalg.norm(s)
    if n < 1e-9:
        s = np.cross(normal, np.array([1., 0., 0.], np.float32))
        n = np.linalg.norm(s)
    return s / (n + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# Debug image
# ═══════════════════════════════════════════════════════════════════════════════

def build_debug(mask_rgb, gmask, depth_s,
                normal, forward, side,
                tang2d, mean_px, cc, rr,
                arrow_scale, H, W):

    left = mask_rgb[:,:,:3].copy()
    hi = np.zeros_like(left); hi[gmask > 0] = (0, 180, 60)
    left = cv2.addWeighted(left, 0.6, hi, 0.4, 0)
    cnts, _ = cv2.findContours(gmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, cnts, -1, (0, 255, 100), 2)

    # White centreline
    if cc is not None:
        for i in range(1, len(cc)):
            cv2.line(left, (int(cc[i-1]),int(rr[i-1])), (int(cc[i]),int(rr[i])),
                     (255,255,255), 1, cv2.LINE_AA)

    # Faint PCA axis
    if tang2d is not None and mean_px is not None:
        ext = max(H, W) * 0.65
        p1 = (int(mean_px[0]-tang2d[0]*ext), int(mean_px[1]-tang2d[1]*ext))
        p2 = (int(mean_px[0]+tang2d[0]*ext), int(mean_px[1]+tang2d[1]*ext))
        cv2.line(left, p1, p2, (160,160,50), 1, cv2.LINE_AA)

    ys, xs = np.where(gmask > 0)
    cx, cy = int(xs.mean()), int(ys.mean())
    cv2.circle(left, (cx,cy), 6, (255,255,255), -1)
    cv2.circle(left, (cx,cy), 8, (0,0,0), 2)

    draw_arrow(left, cx, cy, forward, (255, 80,   0), "forward", arrow_scale, thick=3)
    draw_arrow(left, cx, cy, side,    (  0,120, 255), "side",    arrow_scale, thick=2)
    draw_arrow(left, cx, cy, normal,  (255,255,   0), "normal",  arrow_scale, thick=2)

    def fmt(n, v): return f"{n}: ({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"
    entries = [(fmt("fwd",forward),(255,130,0)),(fmt("sid",side),(0,160,255)),(fmt("nrm",normal),(255,255,0))]
    base_y = H - len(entries)*22 - 12
    for i,(txt,col) in enumerate(entries):
        y = base_y + i*22 + 18
        cv2.putText(left, txt, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(left, txt, (8,y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col,     1, cv2.LINE_AA)

    cv2.putText(left, "side=cross(nrm,fwd) 3D — angle!=90 in image is correct", (8,H-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (130,130,130), 1)
    cv2.putText(left, "Road Orientation V2", (8,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(left, "Road Orientation V2", (8,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    right = cv2.applyColorMap((depth_s*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.drawContours(right, cnts, -1, (0,255,80), 2)
    if cc is not None:
        for i in range(1, len(cc)):
            cv2.line(right, (int(cc[i-1]),int(rr[i-1])), (int(cc[i]),int(rr[i])),
                     (255,255,255), 1, cv2.LINE_AA)
    draw_arrow(right, cx, cy, forward, (255,100, 50), "fwd", arrow_scale, thick=3)
    draw_arrow(right, cx, cy, side,    (  0,120,255), "sid", arrow_scale, thick=2)
    draw_arrow(right, cx, cy, normal,  ( 50,255,255), "nrm", arrow_scale, thick=2)
    cv2.putText(right, "Depth map", (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
    cv2.putText(right, "Depth map", (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    combined = np.concatenate([left[:,:,:3], right], axis=1)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI Node
# ═══════════════════════════════════════════════════════════════════════════════

class RoadOrientationEstimator:
    CATEGORY = "image/analysis"
    FUNCTION = "estimate"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("debug_image", "orientation")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edge_mask_image": ("IMAGE",),
                "depth_map_image": ("IMAGE",),
            },
            "optional": {
                "green_threshold":        ("INT",   {"default":80,   "min":10,  "max":200,   "step":5}),
                "depth_smoothing_kernel": ("INT",   {"default":15,   "min":1,   "max":101,   "step":2}),
                "subsample_step":         ("INT",   {"default":4,    "min":1,   "max":20,    "step":1}),
                "arrow_scale":            ("FLOAT", {"default":120., "min":20., "max":500.,  "step":10.}),
                "min_mask_pixels":        ("INT",   {"default":500,  "min":50,  "max":50000, "step":50}),
            }
        }

    def estimate(self, edge_mask_image, depth_map_image,
                 green_threshold=80, depth_smoothing_kernel=15,
                 subsample_step=4, arrow_scale=120., min_mask_pixels=500):

        mask_rgb  = t2np(edge_mask_image)
        depth_raw = t2np(depth_map_image)
        H, W = mask_rgb.shape[:2]

        if depth_raw.shape[:2] != (H, W):
            depth_raw = cv2.resize(depth_raw, (W, H), interpolation=cv2.INTER_LINEAR)

        gmask = extract_green_mask(mask_rgb, green_threshold)
        gmask = keep_largest_cc(gmask)
        n_px  = int(gmask.sum() // 255)
        if n_px < min_mask_pixels:
            return self._fail(fail_img(H, W, f"Not enough green pixels ({n_px} < {min_mask_pixels})."))

        depth = load_depth(depth_raw)
        depth = smooth_depth(depth, depth_smoothing_kernel)

        # ── STEP 1: Normal (SVD on 3D point cloud, uses depth) ────────────────
        pts = build_pts3d(gmask, depth, subsample_step)
        if len(pts) < 10:
            return self._fail(fail_img(H, W, "Too few 3D points for SVD."))
        normal = svd_normal(pts)
        if normal is None:
            return self._fail(fail_img(H, W, "SVD plane fitting failed."))

        # ── STEP 2: Forward (centreline PCA, Z=0 before plane projection) ─────
        result = centreline_pca_forward(gmask, H, W)
        fwd_raw, tang2d, mean_px, cc, rr = result
        if fwd_raw is None:
            return self._fail(fail_img(H, W, "Centreline PCA failed."))

        # Project onto road plane → forward lies in plane, gets correct Z
        forward = fwd_raw - np.dot(fwd_raw, normal) * normal
        nf = np.linalg.norm(forward)
        if nf < 1e-9:
            return self._fail(fail_img(H, W, "Forward collapsed after plane projection."))
        forward /= nf

        # ── STEP 3: Side (3D cross product) ───────────────────────────────────
        side = cross_side(normal, forward)

        debug = build_debug(mask_rgb, gmask, depth,
                            normal, forward, side,
                            tang2d, mean_px, cc, rr,
                            arrow_scale, H, W)

        return (np2t(debug), self._orientation_json(forward, side, normal))

    # ------------------------------------------------------------------

    def _orientation_json(self, forward, side, normal):
        """
        Pack the full road orientation into a single JSON string.

        The rotation matrix R has the three basis vectors as COLUMNS:
          col 0 = side    (right direction along road surface)
          col 1 = normal  (up, perpendicular to road surface)
          col 2 = forward (forward direction along road)

        To orient a vehicle mesh:
          - Place the mesh so its local X → side, Y → normal, Z → forward
          - Apply R as a 3×3 rotation matrix (or use the 4×4 version
            with identity translation for direct use in most 3D engines)

        All vectors are unit length and mutually orthogonal in 3D.
        """
        import json
        f = [round(float(x), 6) for x in forward]
        s = [round(float(x), 6) for x in side]
        n = [round(float(x), 6) for x in normal]

        # Row-major 3×3: R[row][col], columns = [side, normal, forward]
        rot3 = [
            [s[0], n[0], f[0]],
            [s[1], n[1], f[1]],
            [s[2], n[2], f[2]],
        ]

        # Row-major 4×4 with identity translation (ready for most 3D engines)
        rot4 = [
            [s[0], n[0], f[0], 0.0],
            [s[1], n[1], f[1], 0.0],
            [s[2], n[2], f[2], 0.0],
            [0.0,  0.0,  0.0,  1.0],
        ]

        data = {
            "forward":          f,
            "side":             s,
            "normal":           n,
            "rotation_3x3":     rot3,
            "rotation_4x4":     rot4,
        }
        return json.dumps(data, separators=(',', ':'))

    def _fail(self, img_np):
        import json
        empty = json.dumps({"forward":[0,0,0],"side":[0,0,0],"normal":[0,0,0],
                            "rotation_3x3":[[0]*3]*3,"rotation_4x4":[[0]*4]*4})
        return (np2t(img_np), empty)


NODE_CLASS_MAPPINGS        = {"RoadOrientationEstimator": RoadOrientationEstimator}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadOrientationEstimator": "Road Orientation Estimator V2"}
