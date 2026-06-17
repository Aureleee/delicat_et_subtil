"""
Shared helpers for the DeepLSD road nodes.

LINE_SEGMENTS object format (passed between nodes):
    {
        "lines":  np.ndarray, shape (N, 2, 2), dtype float32
                   each line = [[x0, y0], [x1, y1]] in pixel coordinates
        "width":  int   (image width  the lines were detected on)
        "height": int   (image height the lines were detected on)
    }

This is the "exploitable" output: a plain numpy array you can index, save,
filter, etc.
"""

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:                    # pragma: no cover - torch always present in ComfyUI
    _HAS_TORCH = False

import cv2


# --------------------------------------------------------------------------- #
# LINE_SEGMENTS construction / validation
# --------------------------------------------------------------------------- #
def make_line_segments(lines, width, height):
    """Build a normalised LINE_SEGMENTS dict from any (N,2,2)-like array."""
    arr = np.asarray(lines, dtype=np.float32)
    if arr.size == 0:
        arr = np.zeros((0, 2, 2), dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (2, 2):
        raise ValueError(
            f"lines must have shape (N, 2, 2), got {arr.shape}"
        )
    return {"lines": arr, "width": int(width), "height": int(height)}


def empty_line_segments(width, height):
    return make_line_segments(np.zeros((0, 2, 2), np.float32), width, height)


def get_lines(line_segments):
    """Return the (N,2,2) float32 array from a LINE_SEGMENTS dict (or array)."""
    if isinstance(line_segments, dict):
        return np.asarray(line_segments["lines"], dtype=np.float32)
    return np.asarray(line_segments, dtype=np.float32)


# --------------------------------------------------------------------------- #
# ComfyUI IMAGE <-> numpy
# ComfyUI IMAGE tensors are torch float32, shape (B, H, W, C), range [0, 1], RGB.
# --------------------------------------------------------------------------- #
def comfy_image_to_numpy(image, index=0):
    """ComfyUI IMAGE -> uint8 RGB HxWx3 numpy array (single frame)."""
    if _HAS_TORCH and isinstance(image, torch.Tensor):
        img = image[index].detach().cpu().numpy()
    else:
        img = np.asarray(image)
        if img.ndim == 4:
            img = img[index]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.shape[-1] == 4:                      # drop alpha
        img = img[..., :3]
    return img


def numpy_to_comfy_image(arr):
    """uint8/float RGB HxWx3 numpy array -> ComfyUI IMAGE tensor (1,H,W,3)."""
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    else:
        a = a.astype(np.float32)
        if a.max() > 1.0:
            a = a / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]
    a = a[None, ...]                            # add batch dim
    if _HAS_TORCH:
        return torch.from_numpy(np.ascontiguousarray(a))
    return a


def mask_to_bool(image, index=0, channel_threshold=10):
    """
    Turn a ComfyUI IMAGE used as a mask into a boolean road region.

    The road mask is assumed to be a coloured region on black (green by default).
    Any pixel whose max channel value is above `channel_threshold` (0-255) is
    treated as "inside the region". This is colour-agnostic, so a green, white
    or red mask all work.
    """
    rgb = comfy_image_to_numpy(image, index)
    return rgb.max(axis=-1) > channel_threshold


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def line_length(line):
    (x0, y0), (x1, y1) = line
    return float(np.hypot(x1 - x0, y1 - y0))


def sample_points_on_line(line, n):
    """Return n evenly spaced (x,y) points along the segment, shape (n,2)."""
    (x0, y0), (x1, y1) = line
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    return p0[None, :] * (1 - t) + p1[None, :] * t


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_lines(canvas, lines, color, thickness=2):
    """Draw (N,2,2) lines on a HxWx3 uint8 BGR/RGB canvas (in place)."""
    for (x0, y0), (x1, y1) in np.asarray(lines, dtype=np.float32):
        cv2.line(
            canvas,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            color, thickness, cv2.LINE_AA,
        )
    return canvas
