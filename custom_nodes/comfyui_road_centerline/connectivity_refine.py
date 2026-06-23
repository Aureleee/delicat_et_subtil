"""
Node 6 — RoadConnectivityRefine
=================================
Améliore la connectivité du réseau routier (Wei et al. 2020, section 3.5, Figure 7).

Algorithme (Fig. 7 du papier) :
  Pour chaque intersection détectée :
    1. Chercher les segments de route dont l'extrémité est à moins de max_gap px
    2. Si l'intersection est sur le prolongement du segment (dans un cône angulaire)
       → connecter l'extrémité du fragment à l'intersection
  Résultat : des fragments discontinus sont reliés via les intersections.

En plus :
  - Connexion directe endpoint-to-endpoint si deux paths se terminent
    à moins de gap_direct px l'un de l'autre (sans intersection intermédiaire)
"""

import numpy as np
import cv2
import torch

ROAD_PATHS = "ROAD_PATHS"


def _endpoint_direction(path, from_start=True, n_pts=5):
    """Estime la direction à l'extrémité d'un path (vecteur unitaire x,y)."""
    pts = path["points"]  # (N,2) xy
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    if from_start:
        a = pts[min(n_pts - 1, len(pts) - 1)]
        b = pts[0]
    else:
        a = pts[max(len(pts) - n_pts, 0)]
        b = pts[-1]
    d = b - a
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([1.0, 0.0])


def _extend_to_point(path, target_xy, from_start=True):
    """Étend un path depuis une extrémité vers target_xy (ajoute le point)."""
    pts    = path["points"].tolist()
    widths = path["widths"].tolist()
    angles = path["angles"].tolist()

    tx, ty = float(target_xy[0]), float(target_xy[1])
    # Direction d'extension
    ref = np.array(pts[0] if from_start else pts[-1])
    d   = np.array([tx, ty]) - ref
    ang = float(np.arctan2(d[1], d[0]))
    w   = float(widths[0] if from_start else widths[-1])

    if from_start:
        pts.insert(0, [tx, ty])
        widths.insert(0, w)
        angles.insert(0, ang)
    else:
        pts.append([tx, ty])
        widths.append(w)
        angles.append(ang)

    return {
        "points": np.array(pts, dtype=np.float32),
        "widths": np.array(widths, dtype=np.float32),
        "angles": np.array(angles, dtype=np.float32),
    }


class RoadConnectivityRefine:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = (ROAD_PATHS, "INT", "IMAGE")
    RETURN_NAMES  = ("refined_paths", "n_paths", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "road_paths":        (ROAD_PATHS,),
                "intersection_mask": ("MASK",),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "max_gap_px": ("INT", {
                    "default": 40, "min": 5, "max": 200,
                    "tooltip": "Distance max (pixels) entre une extrémité de path et une "
                               "intersection pour connecter les deux."}),
                "angle_cone_deg": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Cône angulaire (demi-angle) : le prolongement du segment "
                               "doit pointer vers l'intersection dans ce cône."}),
                "gap_direct_px": ("INT", {
                    "default": 25, "min": 0, "max": 100,
                    "tooltip": "Distance max pour connecter directement deux endpoints "
                               "sans intersection intermédiaire. 0 = désactivé."}),
            }
        }

    def run(self, road_paths, intersection_mask,
            background_image=None, max_gap_px=40,
            angle_cone_deg=30.0, gap_direct_px=25):

        im = intersection_mask[0] if intersection_mask.dim() == 3 else intersection_mask
        inter_np  = (im.cpu().numpy() > 0.5)
        inter_pts = np.argwhere(inter_np)  # (K, 2) yx

        H = inter_np.shape[0]; W = inter_np.shape[1]
        paths = [dict(p) for p in (road_paths or [])]   # copie
        cone_rad = np.radians(angle_cone_deg)

        # ── Connexion via intersections ────────────────────────────────────────
        for k, path in enumerate(paths):
            pts = path["points"]  # (N,2) xy
            if len(pts) < 2 or len(inter_pts) == 0:
                continue

            for from_start in [True, False]:
                ep_xy = pts[0] if from_start else pts[-1]
                ep_dir = _endpoint_direction(path, from_start=from_start)

                # Chercher l'intersection la plus proche dans le cône angulaire
                best_dist = max_gap_px + 1
                best_ipt  = None

                for (iy, ix) in inter_pts:
                    ipt = np.array([float(ix), float(iy)])
                    d   = ipt - ep_xy
                    dist_i = float(np.linalg.norm(d))
                    if dist_i < 1.0 or dist_i > max_gap_px:
                        continue
                    d_norm = d / dist_i
                    cos_a  = float(np.dot(ep_dir, d_norm))
                    if cos_a < np.cos(cone_rad):
                        continue
                    if dist_i < best_dist:
                        best_dist = dist_i
                        best_ipt  = ipt

                if best_ipt is not None:
                    paths[k] = _extend_to_point(path, best_ipt, from_start)

        # ── Connexion directe endpoint-to-endpoint ────────────────────────────
        if gap_direct_px > 0 and len(paths) > 1:
            merged = [False] * len(paths)
            result = []
            for i in range(len(paths)):
                if merged[i]:
                    continue
                pi = paths[i]
                best_j = -1; best_d = gap_direct_px + 1
                best_config = None  # (from_start_i, from_start_j)

                for j in range(i + 1, len(paths)):
                    if merged[j]:
                        continue
                    pj = paths[j]
                    for fs_i in [True, False]:
                        ep_i = pi["points"][0] if fs_i else pi["points"][-1]
                        for fs_j in [True, False]:
                            ep_j = pj["points"][0] if fs_j else pj["points"][-1]
                            d = float(np.linalg.norm(ep_i - ep_j))
                            if d < best_d:
                                best_d = d; best_j = j
                                best_config = (fs_i, fs_j)

                if best_j >= 0:
                    fs_i, fs_j = best_config
                    # Fusionner pj dans pi
                    pj = paths[best_j]
                    pts_j = pj["points"] if not fs_j else pj["points"][::-1]
                    ws_j  = pj["widths"] if not fs_j else pj["widths"][::-1]
                    an_j  = pj["angles"] if not fs_j else pj["angles"][::-1]

                    if fs_i:
                        pts_m = np.vstack([pts_j, pi["points"]])
                        ws_m  = np.concatenate([ws_j,  pi["widths"]])
                        an_m  = np.concatenate([an_j,  pi["angles"]])
                    else:
                        pts_m = np.vstack([pi["points"], pts_j])
                        ws_m  = np.concatenate([pi["widths"],  ws_j])
                        an_m  = np.concatenate([pi["angles"],  an_j])

                    paths[i] = {"points": pts_m, "widths": ws_m, "angles": an_m}
                    merged[best_j] = True

                result.append(paths[i])
            paths = result

        # ── Debug image ───────────────────────────────────────────────────────
        if background_image is not None:
            bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), dtype=np.uint8)

        colors_bgr = [
            (0, 200, 255), (0, 255, 100), (255, 100, 0),
            (200, 0, 255), (255, 200, 0), (0, 100, 255),
        ]
        for i, path in enumerate(paths):
            col  = colors_bgr[i % len(colors_bgr)]
            arr  = path["points"].astype(np.int32)
            arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
            arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
            for a, b in zip(arr[:-1], arr[1:]):
                cv2.line(debug, (a[0], a[1]), (b[0], b[1]), col, 2, cv2.LINE_AA)
            cv2.circle(debug, (arr[0, 0],  arr[0, 1]),  6, (0, 255, 255), -1)
            cv2.circle(debug, (arr[-1, 0], arr[-1, 1]), 6, (0, 180, 255), -1)

        # Intersections en rouge
        for (iy, ix) in inter_pts:
            cv2.circle(debug, (int(ix), int(iy)), 6, (0, 0, 255), 2)

        cv2.putText(debug, f"{len(paths)} refined paths",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        debug_t   = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (paths, len(paths), debug_t)


NODE_CLASS_MAPPINGS        = {"RoadConnectivityRefine": RoadConnectivityRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadConnectivityRefine": "Road Refine"}
