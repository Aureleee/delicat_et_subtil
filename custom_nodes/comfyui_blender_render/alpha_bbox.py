"""
AlphaBBox
=========
Prend une image RGBA (tank sur fond transparent) et retourne la bounding box
serrée autour des pixels non-transparents.

Outputs :
  debug_image  : image RGBA avec la bbox dessinée (vert)
  x            : pixel gauche
  y            : pixel haut
  width        : largeur en pixels
  height       : hauteur en pixels
  cx           : centre X
  cy           : centre Y
"""

import numpy as np
import torch
import cv2


def _t2np_rgba(t):
    """Tensor (1,H,W,4) ou (1,H,W,3) → np.uint8 RGBA."""
    if t.dim() == 4:
        t = t[0]
    arr = (t.detach().cpu().float().numpy().clip(0, 1) * 255).astype(np.uint8)
    if arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    return arr

def _np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


class AlphaBBox:
    CATEGORY = "blender/utils"
    FUNCTION = "compute"
    RETURN_TYPES  = ("IMAGE", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES  = ("debug_image", "x", "y", "width", "height", "cx", "cy")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha_threshold": ("INT", {
                    "default": 10, "min": 0, "max": 254, "step": 1,
                    "tooltip": "Seuil alpha : pixels avec alpha > seuil = tank."}),
                "box_color_r": ("INT", {"default": 0,   "min": 0, "max": 255}),
                "box_color_g": ("INT", {"default": 255, "min": 0, "max": 255}),
                "box_color_b": ("INT", {"default": 80,  "min": 0, "max": 255}),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 10}),
            }
        }

    def compute(self, image, alpha_threshold=10,
                box_color_r=0, box_color_g=255, box_color_b=80,
                line_thickness=2):

        rgba = _t2np_rgba(image)
        H, W = rgba.shape[:2]

        alpha = rgba[:, :, 3]
        mask  = (alpha > alpha_threshold).astype(np.uint8)

        if mask.sum() == 0:
            vis = rgba.copy()
            cv2.putText(vis, "No opaque pixels found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 80, 80, 255), 2)
            return (_np2t(vis), 0, 0, W, H, W // 2, H // 2)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y0, y1 = int(np.argmax(rows)), int(H - 1 - np.argmax(rows[::-1]))
        x0, x1 = int(np.argmax(cols)), int(W - 1 - np.argmax(cols[::-1]))

        bx, by   = x0, y0
        bw, bh   = x1 - x0 + 1, y1 - y0 + 1
        bcx, bcy = x0 + bw // 2, y0 + bh // 2

        # Debug : fond checkerboard gris pour voir la transparence, bbox verte
        checker = np.zeros((H, W, 3), np.uint8)
        cs = 16
        for r in range(0, H, cs):
            for c in range(0, W, cs):
                v = 200 if ((r // cs + c // cs) % 2 == 0) else 160
                checker[r:r+cs, c:c+cs] = v

        rgb   = rgba[:, :, :3].astype(np.float32)
        a_f   = (alpha.astype(np.float32) / 255.0)[:, :, None]
        blended = (rgb * a_f + checker.astype(np.float32) * (1 - a_f)).clip(0, 255).astype(np.uint8)
        vis = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        color_bgr = (box_color_b, box_color_g, box_color_r)
        cv2.rectangle(vis, (bx, by), (bx + bw - 1, by + bh - 1), color_bgr, line_thickness, cv2.LINE_AA)
        cv2.circle(vis, (bcx, bcy), 4, color_bgr, -1, cv2.LINE_AA)

        label = f"bbox  x={bx} y={by}  {bw}x{bh}px"
        cv2.putText(vis, label, (bx, max(by - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return (_np2t(vis_rgb), bx, by, bw, bh, bcx, bcy)


class TankWidthSides:
    """
    Trouve les deux côtés LONGS du tank (= les flancs, parallèles à la longueur).
    La distance perpendiculaire entre ces deux côtés = vraie largeur du tank.
    Utilise le rectangle minimal orienté (minAreaRect) sur les pixels opaques.
    """
    CATEGORY = "blender/utils"
    FUNCTION = "compute"
    RETURN_TYPES  = ("IMAGE", "FLOAT")
    RETURN_NAMES  = ("debug_image", "tank_width_px")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "alpha_threshold": ("INT", {"default": 1, "min": 0, "max": 254}),
            }
        }

    def compute(self, image, alpha_threshold=1):
        rgba = _t2np_rgba(image)
        H, W = rgba.shape[:2]
        alpha = rgba[:, :, 3]

        ys, xs = np.where(alpha > alpha_threshold)
        if len(xs) < 5:
            rgb = rgba[:, :, :3].copy()
            return (_np2t(rgb), 0.0)

        points = np.column_stack([xs, ys]).astype(np.float32)

        # Rectangle minimal orienté autour des pixels opaques
        rect  = cv2.minAreaRect(points)
        box   = cv2.boxPoints(rect)   # 4 coins dans l'ordre

        # Identifier les deux côtés longs (= flancs du tank)
        side_01 = float(np.linalg.norm(box[1] - box[0]))
        side_12 = float(np.linalg.norm(box[2] - box[1]))

        if side_01 >= side_12:
            long_sides  = [(box[0], box[1]), (box[3], box[2])]
            tank_width  = side_12
        else:
            long_sides  = [(box[1], box[2]), (box[0], box[3])]
            tank_width  = side_01

        # Fond checkerboard + tank RGB
        checker = np.zeros((H, W, 3), np.uint8)
        cs = 16
        for r in range(0, H, cs):
            for c in range(0, W, cs):
                v = 200 if ((r // cs + c // cs) % 2 == 0) else 160
                checker[r:r+cs, c:c+cs] = v
        rgb = rgba[:, :, :3].astype(np.float32)
        a_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
        blended = (rgb * a_f + checker.astype(np.float32) * (1 - a_f)).clip(0, 255).astype(np.uint8)
        vis = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

        # Dessiner les deux côtés longs en jaune vif
        for (p0, p1) in long_sides:
            cv2.line(vis,
                     (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(p1[0])), int(round(p1[1]))),
                     (0, 220, 255), 2, cv2.LINE_AA)

        # Dessiner la flèche de largeur (entre milieux des deux côtés longs)
        mid0 = ((long_sides[0][0] + long_sides[0][1]) / 2).astype(int)
        mid1 = ((long_sides[1][0] + long_sides[1][1]) / 2).astype(int)
        cv2.arrowedLine(vis, tuple(mid0), tuple(mid1), (0, 255, 128), 2, cv2.LINE_AA, tipLength=0.1)
        cv2.arrowedLine(vis, tuple(mid1), tuple(mid0), (0, 255, 128), 2, cv2.LINE_AA, tipLength=0.1)

        label = f"tank_width = {tank_width:.1f} px"
        cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        return (_np2t(vis_rgb), float(tank_width))


NODE_CLASS_MAPPINGS        = {"AlphaBBox": AlphaBBox, "TankWidthSides": TankWidthSides}
NODE_DISPLAY_NAME_MAPPINGS = {"AlphaBBox": "Alpha BBox (tank)", "TankWidthSides": "Tank Width Sides"}
