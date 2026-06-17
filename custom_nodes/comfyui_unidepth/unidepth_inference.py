"""
UniDepth Inference Node
=======================
Lance l'inférence UniDepth sur une image et retourne :
  - depth_map      : carte de profondeur en mètres (IMAGE, colorisée magma)
  - depth_raw      : IMAGE float [0-1] normalisée de la depth (pour debug)
  - metric_depth   : tenseur 1xHxW en MÈTRES (UNIDEPTH_DEPTH)
  - point_cloud    : tenseur 3xHxW XYZ caméra (UNIDEPTH_POINTS)
  - intrinsics     : tenseur 3x3 (UNIDEPTH_INTRINSICS)
  - confidence_map : IMAGE colorisée (V2 uniquement, sinon blanc)
  - summary        : STRING avec min/max/median depth + infos

Types custom exportés :
  UNIDEPTH_DEPTH       : torch.Tensor (1, H, W) float32, en mètres
  UNIDEPTH_POINTS      : torch.Tensor (3, H, W) float32, XYZ caméra (mètres)
  UNIDEPTH_INTRINSICS  : torch.Tensor (3, 3) float32
"""

import numpy as np
import torch
import cv2
from .unidepth_loader import UNIDEPTH_MODEL

UNIDEPTH_DEPTH      = "UNIDEPTH_DEPTH"
UNIDEPTH_POINTS     = "UNIDEPTH_POINTS"
UNIDEPTH_INTRINSICS = "UNIDEPTH_INTRINSICS"


def _t2np(t):
    if t.dim() == 4:
        t = t[0]
    return (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)


def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


def _colorize_depth(depth_np, vmin=None, vmax=None, cmap=cv2.COLORMAP_MAGMA):
    """depth_np : (H, W) float32 en mètres → (H, W, 3) uint8 RGB colorisé."""
    valid = depth_np[depth_np > 0]
    if vmin is None:
        vmin = float(valid.min()) if valid.size > 0 else 0.0
    if vmax is None:
        vmax = float(valid.max()) if valid.size > 0 else 1.0
    norm = np.clip((depth_np - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cmap)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _colorize_confidence(conf_np):
    """conf_np : (H, W) float32 [0,1] → (H, W, 3) uint8."""
    norm = np.clip(conf_np, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_RdYlGn if False else cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


class UniDepthInference:
    """
    Lance UniDepth sur une image RGB.

    Outputs :
      depth_map      : carte de profondeur colorisée (IMAGE, colormap magma)
      confidence_map : carte de confiance colorisée (IMAGE, V2 only — sinon blanc)
      metric_depth   : UNIDEPTH_DEPTH  tenseur (1,H,W) en MÈTRES
      point_cloud    : UNIDEPTH_POINTS tenseur (3,H,W) XYZ en coordonnées caméra (mètres)
      intrinsics     : UNIDEPTH_INTRINSICS tenseur (3,3) — fx,fy,cx,cy estimés
      summary        : informations de profondeur (min/max/median/fov estimé)

    Note sur la confidence (V2) :
      UniDepthV2 prédit une confidence par pixel. Une faible confidence indique
      que le modèle est incertain sur la profondeur à cet endroit (typiquement :
      zones sans texture, ciel, reflets, bords d'objets, objets transparents).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "unidepth_model":  (UNIDEPTH_MODEL,),
            },
            "optional": {
                "depth_vmin_m": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 200.0, "step": 0.5,
                    "tooltip": "Min depth pour la colorisation (0 = auto)"}),
                "depth_vmax_m": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 300.0, "step": 1.0,
                    "tooltip": "Max depth pour la colorisation (0 = auto)"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", UNIDEPTH_DEPTH, UNIDEPTH_POINTS,
                     UNIDEPTH_INTRINSICS, "STRING")
    RETURN_NAMES  = ("depth_map", "confidence_map", "metric_depth",
                     "point_cloud", "intrinsics", "summary")
    FUNCTION      = "infer"
    CATEGORY      = "depth/unidepth"

    def infer(self, image, unidepth_model, depth_vmin_m=0.0, depth_vmax_m=0.0):
        model = unidepth_model["model"]
        info  = unidepth_model["info"]
        device = next(model.parameters()).device

        # ── Préparer l'image ──────────────────────────────────────────────────
        # image ComfyUI : (1, H, W, 3) float [0,1]
        rgb_np = (image[0].cpu().float().numpy() * 255).astype(np.uint8)  # H,W,3
        H, W = rgb_np.shape[:2]
        rgb_torch = torch.from_numpy(rgb_np).permute(2, 0, 1)  # 3,H,W uint8

        # ── Inférence ─────────────────────────────────────────────────────────
        with torch.no_grad():
            predictions = model.infer(rgb_torch)

        # ── Extraire les outputs ───────────────────────────────────────────────
        # depth : (1, 1, H, W) en mètres
        depth_t = predictions["depth"].squeeze(0)       # (1, H, W)
        depth_np = depth_t.squeeze(0).cpu().float().numpy()  # (H, W)

        # points : (1, 3, H, W) XYZ caméra
        points_t = predictions["points"].squeeze(0)     # (3, H, W)

        # intrinsics : (1, 3, 3) ou (3, 3)
        K_t = predictions["intrinsics"]
        if K_t.dim() == 3:
            K_t = K_t.squeeze(0)   # (3, 3)

        # confidence (V2 only) : (1, 1, H, W) ou None
        conf_np = None
        if "confidence" in predictions and predictions["confidence"] is not None:
            conf_t = predictions["confidence"]
            if conf_t.dim() == 4:
                conf_t = conf_t.squeeze(0).squeeze(0)
            elif conf_t.dim() == 3:
                conf_t = conf_t.squeeze(0)
            conf_np = conf_t.cpu().float().numpy()
            # La confidence est en log-space dans certaines versions, normaliser
            conf_np = np.clip(conf_np, 0, None)
            if conf_np.max() > 1.0:
                conf_np = conf_np / conf_np.max()

        # ── Coloriser depth ────────────────────────────────────────────────────
        valid = depth_np[depth_np > 0.1]
        d_min = float(valid.min()) if valid.size > 0 else 0.0
        d_max = float(valid.max()) if valid.size > 0 else 10.0
        d_med = float(np.median(valid)) if valid.size > 0 else 5.0

        vmin = depth_vmin_m if depth_vmin_m > 0 else d_min
        vmax = depth_vmax_m if depth_vmax_m > 0 else d_max

        depth_colored = _colorize_depth(depth_np, vmin=vmin, vmax=vmax)
        depth_img_t   = _np2t(depth_colored)

        # ── Coloriser confidence ───────────────────────────────────────────────
        if conf_np is not None:
            conf_colored = _colorize_confidence(conf_np)
        else:
            conf_colored = np.ones((H, W, 3), np.uint8) * 200  # gris = N/A
            cv2.putText(conf_colored, "V1: no confidence", (10, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)
        conf_img_t = _np2t(conf_colored)

        # ── Intrinsics infos ───────────────────────────────────────────────────
        K_np = K_t.cpu().float().numpy()
        fx, fy = float(K_np[0, 0]), float(K_np[1, 1])
        cx, cy = float(K_np[0, 2]), float(K_np[1, 2])
        fov_h = float(np.degrees(2 * np.arctan(W / (2 * fx)))) if fx > 0 else 0
        fov_v = float(np.degrees(2 * np.arctan(H / (2 * fy)))) if fy > 0 else 0

        # ── Summary ───────────────────────────────────────────────────────────
        has_conf = conf_np is not None
        conf_mean = float(conf_np.mean()) if has_conf else 0.0
        summary = (
            f"UniDepth {info['version']} | {info['backbone']} | {W}x{H}\n"
            f"Depth  : min={d_min:.2f}m  median={d_med:.2f}m  max={d_max:.2f}m\n"
            f"FoV    : H={fov_h:.1f}deg  V={fov_v:.1f}deg\n"
            f"FX={fx:.1f}  FY={fy:.1f}  CX={cx:.1f}  CY={cy:.1f}\n"
            f"Confiance moyenne : {conf_mean:.2f}" if has_conf
            else f"Confidence : N/A (V1)"
        )

        return (
            depth_img_t,
            conf_img_t,
            depth_t.cpu().float(),          # (1,H,W) mètres
            points_t.cpu().float(),         # (3,H,W) XYZ mètres
            K_t.cpu().float(),              # (3,3)
            summary,
        )


NODE_CLASS_MAPPINGS        = {"UniDepthInference": UniDepthInference}
NODE_DISPLAY_NAME_MAPPINGS = {"UniDepthInference": "UniDepth Inference"}
