"""
Road Mask Centerline
====================
Prend l'image de masque vert, sélectionne le PLUS GRAND segment vert,
calcule la centreline par régression linéaire, et sort :
  - centerline_image : visualisation du segment + centreline tracée
  - orientation      : JSON compatible RoadOrientationPlotter
                       (normal = placeholder plan flat,
                        forward = direction centreline en espace image)
"""

import json
import numpy as np
import cv2
import torch


# ═══════════════════════════════════════════════════════════════════════════════
def t2np(t):
    if t.dim() == 4: t = t[0]
    return (np.clip(t.detach().cpu().float().numpy(), 0, 1) * 255).astype(np.uint8)

def np2t(a):
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)

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

def compute_centerline_regression(mask, H, W):
    """
    Par ligne : milieu entre px le plus à gauche et le plus à droite.
    Régression linéaire col = a*row + b.
    Retourne (tang2d [dcol,drow] unitaire orienté vers le haut de l'image,
              mean_pt [col,row], cc, rr) ou (None,None,None,None).
    """
    cc, rr = [], []
    for row in range(H):
        px = np.where(mask[row] > 0)[0]
        if len(px) < 2: continue
        cc.append((float(px.min()) + float(px.max())) * 0.5)
        rr.append(float(row))
    if len(cc) < 5:
        return None, None, None, None

    cc = np.array(cc, np.float32)
    rr = np.array(rr, np.float32)
    try:
        a, _ = np.polyfit(rr, cc, 1)
    except Exception:
        return None, None, None, None

    tang = np.array([a, 1.0], np.float32)
    tang /= np.linalg.norm(tang) + 1e-9

    # Orienter vers les lignes hautes (l'extrémité "loin" de la caméra)
    mean_r = rr.mean()
    above = rr < mean_r
    below = rr >= mean_r
    if above.any() and below.any():
        # Le vecteur tangent [dcol, drow] avec drow<0 pointe vers le haut → correct
        if tang[1] > 0:      # pointe vers le bas → inverser
            tang = -tang

    mean_pt = np.array([cc.mean(), rr.mean()], np.float32)
    return tang, mean_pt, cc, rr


def pack_orientation(fwd, sid, nrm):
    f = [round(float(x), 6) for x in fwd]
    s = [round(float(x), 6) for x in sid]
    n = [round(float(x), 6) for x in nrm]
    return json.dumps({
        "forward": f, "side": s, "normal": n,
        "rotation_3x3": [[s[0],n[0],f[0]],[s[1],n[1],f[1]],[s[2],n[2],f[2]]],
        "rotation_4x4": [[s[0],n[0],f[0],0.],[s[1],n[1],f[1],0.],
                          [s[2],n[2],f[2],0.],[0.,0.,0.,1.]],
    }, separators=(',',':'))


# ═══════════════════════════════════════════════════════════════════════════════
class RoadMaskCenterline:

    CATEGORY     = "image/analysis"
    FUNCTION     = "analyze"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("centerline_image", "orientation")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),
            },
            "optional": {
                "green_threshold": ("INT", {
                    "default": 80, "min": 10, "max": 200, "step": 5,
                    "tooltip": "Dominance minimale du canal vert pour isoler la route.",
                }),
                "min_segment_pixels": ("INT", {
                    "default": 500, "min": 10, "max": 1000000,
                    "tooltip": "Taille minimale (pixels) du segment pour être considéré.",
                }),
                "camera_elevation_deg": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 89.0, "step": 1.0,
                    "tooltip": "Élévation approximative de la caméra (degrés). "
                               "Sert uniquement à construire la normale placeholder "
                               "pour le plotter. Ne pas confondre avec la vraie normale SVD.",
                }),
            },
        }

    def analyze(self, mask_image,
                green_threshold=80, min_segment_pixels=500,
                camera_elevation_deg=15.0):

        rgb = t2np(mask_image)
        H, W = rgb.shape[:2]

        # ── 1. Masque vert + plus grand segment ──────────────────────────────
        gmask = extract_green_mask(rgb, green_threshold)
        road_pixels = int((gmask > 0).sum())

        if road_pixels < min_segment_pixels:
            # Rien de trouvé : retourner image originale + orientation nulle
            empty = pack_orientation(
                np.array([0.,0.,1.]), np.array([1.,0.,0.]), np.array([0.,-0.3,0.95])
            )
            vis_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb.ndim==3 else rgb
            return (np2t(vis_rgb), empty)

        gmask = keep_largest_cc(gmask)

        # ── 2. Centreline par régression ─────────────────────────────────────
        tang, mean_pt, cc, rr = compute_centerline_regression(gmask, H, W)

        # ── 3. Visualisation ─────────────────────────────────────────────────
        vis = rgb.copy()
        # Colorier le segment retenu
        road_px = gmask > 0
        vis[road_px] = np.clip(
            vis[road_px].astype(np.float32) * 0.45 + np.array([30, 200, 60]) * 0.55,
            0, 255
        ).astype(np.uint8)

        if tang is not None and mean_pt is not None:
            ext = max(H, W) * 0.8
            p1 = (int(mean_pt[0] - tang[0]*ext), int(mean_pt[1] - tang[1]*ext))
            p2 = (int(mean_pt[0] + tang[0]*ext), int(mean_pt[1] + tang[1]*ext))
            cv2.line(vis, p1, p2, (0, 255, 140), 3, cv2.LINE_AA)

            # Points centreline
            step = max(1, len(cc)//60)
            for c_v, r_v in zip(cc[::step], rr[::step]):
                cv2.circle(vis, (int(c_v), int(r_v)), 3, (255, 210, 0), -1, cv2.LINE_AA)

            # Flèche : direction "loin" (vers le haut de l'image)
            mp = (int(mean_pt[0]), int(mean_pt[1]))
            reach = min(H, W) // 5
            ep = (int(mean_pt[0] + tang[0]*reach), int(mean_pt[1] + tang[1]*reach))
            cv2.arrowedLine(vis, mp, ep, (0, 240, 200), 4,
                            tipLength=0.25, line_type=cv2.LINE_AA)

            # Angle affiché
            angle_deg = float(np.degrees(np.arctan2(-tang[0], -tang[1])))  # par rapport à image-up
            cv2.putText(vis, f"Road angle: {angle_deg:.1f} deg",
                        (10, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, f"Road angle: {angle_deg:.1f} deg",
                        (10, H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80,255,160), 1, cv2.LINE_AA)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        # ── 4. Orientation JSON ───────────────────────────────────────────────
        if tang is None:
            fwd = np.array([0., -1., 0.], np.float32)
        else:
            # forward = tangent 2D normalisé, Z=0 (plan image)
            fwd = np.array([tang[0], tang[1], 0.], np.float32)
            fwd /= np.linalg.norm(fwd) + 1e-9

        # Normale placeholder : dépend de l'élévation caméra
        el = float(np.radians(camera_elevation_deg))
        nrm = np.array([0., -float(np.sin(el)), float(np.cos(el))], np.float32)
        nrm /= np.linalg.norm(nrm) + 1e-9

        # Projeter forward sur le plan de la normale, recalculer side
        fwd = fwd - np.dot(fwd, nrm) * nrm
        nf = np.linalg.norm(fwd)
        if nf < 1e-9:
            fwd = np.array([0., 0., 1.], np.float32)
        else:
            fwd /= nf

        sid = np.cross(nrm, fwd).astype(np.float32)
        sid /= np.linalg.norm(sid) + 1e-9
        fwd = np.cross(sid, nrm).astype(np.float32)
        fwd /= np.linalg.norm(fwd) + 1e-9

        orientation_str = pack_orientation(fwd, sid, nrm)
        return (np2t(vis_rgb), orientation_str)


NODE_CLASS_MAPPINGS        = {"RoadMaskCenterline": RoadMaskCenterline}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadMaskCenterline": "Road Mask Centerline"}
