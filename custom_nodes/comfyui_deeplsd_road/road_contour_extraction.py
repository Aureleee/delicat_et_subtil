"""
ComfyUI Custom Node : Road Contour Extraction  (Node 1 / 3)
===========================================================
Extract EVERY visible contour of a road mask -- nothing more.

Goal
----
Recover all points belonging to the contour of the mask, WITHOUT trying to
tell left from right and WITHOUT any correction or reconstruction. Defects of
the mask are kept exactly as they are:
  - holes
  - missing areas
  - cuts
  - several independent roads

Outputs
-------
  - debug_image : every detected contour drawn on a faint mask
                  (outer outlines = green, hole outlines = orange)
  - contours    : ROAD_CONTOURS  (consumed by "Corrupted Border Detection")
  - summary     : STRING

ROAD_CONTOURS object format (passed between the 3 contour nodes):
    {
        "contours": [np.ndarray (Ni, 2) float32, ...]   # ordered outline pts
        "is_hole":  [bool, ...]                          # one per contour
        "reliable": [np.ndarray (Ni,) bool, ...] | None  # filled by Node 2
        "width":  int,
        "height": int,
    }
"""

import numpy as np
import cv2

from .line_helpers import numpy_to_comfy_image, mask_to_bool


def _find_contours(mask_u8):
    """Return (contours, is_hole) keeping holes as inner contours."""
    res = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = res[-2], res[-1]
    out, holes = [], []
    if hierarchy is None:
        return out, holes
    for c, h in zip(contours, hierarchy[0]):
        pts = c.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 2:
            continue
        out.append(pts)
        holes.append(bool(h[3] != -1))          # parent != -1  -> it's a hole
    return out, holes


class RoadContourExtraction:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "road_mask": ("IMAGE",),
            },
            "optional": {
                "mask_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "min_contour_pts": ("INT", {"default": 8, "min": 2, "max": 5000}),
                "min_contour_area": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1e7, "step": 10.0}),
                "keep_holes": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "ROAD_CONTOURS", "STRING")
    RETURN_NAMES = ("debug_image", "contours", "summary")
    FUNCTION = "extract"
    CATEGORY = "DeepLSD Road/3 Contours"

    def extract(self, road_mask, mask_threshold=10, min_contour_pts=8,
                min_contour_area=0.0, keep_holes=True):
        mask_bool = mask_to_bool(road_mask, 0, mask_threshold)
        h, w = mask_bool.shape
        mask_u8 = mask_bool.astype(np.uint8)

        contours, is_hole = _find_contours(mask_u8)

        kept, kept_hole = [], []
        for pts, hole in zip(contours, is_hole):
            if hole and not keep_holes:
                continue
            if pts.shape[0] < min_contour_pts:
                continue
            if min_contour_area > 0.0:
                area = abs(cv2.contourArea(pts.astype(np.float32)))
                if area < min_contour_area:
                    continue
            kept.append(pts)
            kept_hole.append(hole)

        # --- debug image ---
        debug = np.zeros((h, w, 3), np.uint8)
        debug[mask_bool] = (35, 60, 35)                      # faint road
        for pts, hole in zip(kept, kept_hole):
            ip = np.round(pts).astype(np.int32)
            col = (255, 150, 0) if hole else (0, 220, 70)    # orange / green
            cv2.polylines(debug, [ip], isClosed=True, color=col,
                          thickness=2, lineType=cv2.LINE_AA)

        out = {
            "contours": kept,
            "is_hole": kept_hole,
            "reliable": None,
            "width": int(w),
            "height": int(h),
        }
        n_out = sum(1 for x in kept_hole if not x)
        n_hole = sum(1 for x in kept_hole if x)
        total_pts = int(sum(p.shape[0] for p in kept))
        summary = (f"Contours: {len(kept)} total "
                   f"({n_out} outer, {n_hole} hole) | {total_pts} points. "
                   f"No reconstruction applied.")
        return (numpy_to_comfy_image(debug), out, summary)
