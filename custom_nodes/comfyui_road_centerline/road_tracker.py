"""
Node 3 — RoadTracker
=====================
Tracking des centerlines (Wei et al. 2020, section 3.4, Eq. 8-11).

Algorithme du papier :
  - Sélectionner un point de départ (endpoint du squelette)
  - Direction initiale : gradient local de la confidence map
  - Candidats suivants : xs,t = x + cos(θ+t)*S,  ys,t = y + sin(θ+t)*S
    avec t ∈ {0°, ±1°, ..., ±10°} et S = step_size (défaut 15px)
  - Choisir argmin|t| tel que C(xs,t, ys,t) == 1 (sur centerline)
  - Mettre à jour θ ← θ + t
  - Répéter jusqu'à sortie de la centerline

Pour chaque path : récupère les largeurs depuis la width_map.

Type de sortie : ROAD_PATHS = liste de dicts
  {"points": (N,2) float32 xy,
   "widths":  (N,)  float32,
   "angles":  (N,)  float32 (radians)}
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

ROAD_PATHS = "ROAD_PATHS"


def _initial_direction(y, x, conf, H, W):
    """Direction initiale en (x,y) : gradient de la confidence map."""
    r = 3
    y0, y1 = max(0, y - r), min(H - 1, y + r)
    x0, x1 = max(0, x - r), min(W - 1, x + r)
    patch = conf[y0:y1+1, x0:x1+1]
    Dx = cv2.Sobel(patch.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    Dy = cv2.Sobel(patch.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    cy, cx = (y - y0), (x - x0)
    cy = min(cy, Dx.shape[0] - 1); cx = min(cx, Dx.shape[1] - 1)
    dx = float(Dx[cy, cx]); dy = float(Dy[cy, cx])
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    # On veut la direction tangente (perpendiculaire au gradient)
    return float(np.arctan2(dx, -dy))


def _track_one(start_y, start_x, start_theta, centerline, conf, width_map,
               visited, H, W, step_size=15, max_angle_deg=10, max_steps=500):
    """Suit une centerline depuis (start_x, start_y) dans la direction start_theta."""
    angle_range = list(range(0, max_angle_deg + 1))
    t_candidates = [0]
    for a in angle_range[1:]:
        t_candidates += [a, -a]

    path_pts  = [(start_x, start_y)]
    path_w    = [float(width_map[start_y, start_x]) if 0 <= start_y < H and 0 <= start_x < W else 1.0]
    path_ang  = [start_theta]
    theta     = start_theta

    # Marquer un voisinage autour du point de départ comme visité
    for dy in range(-step_size // 2, step_size // 2 + 1):
        for dx in range(-step_size // 2, step_size // 2 + 1):
            ny2 = start_y + dy; nx2 = start_x + dx
            if 0 <= ny2 < H and 0 <= nx2 < W:
                visited.add((ny2, nx2))

    for _ in range(max_steps):
        cx, cy = path_pts[-1]
        found  = False

        for t_deg in t_candidates:
            t_rad   = np.radians(t_deg)
            new_dir = theta + t_rad
            nx = cx + np.cos(new_dir) * step_size
            ny = cy + np.sin(new_dir) * step_size
            ni = int(round(ny)); nj = int(round(nx))

            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if (ni, nj) in visited:
                continue
            if centerline[ni, nj] == 0:
                continue

            theta = new_dir
            # Marquer le voisinage du nouveau point comme visité
            half = step_size // 2
            for dy2 in range(-half, half + 1):
                for dx2 in range(-half, half + 1):
                    vy = ni + dy2; vx = nj + dx2
                    if 0 <= vy < H and 0 <= vx < W:
                        visited.add((vy, vx))
            path_pts.append((nj, ni))
            path_w.append(float(width_map[ni, nj]) if width_map is not None else 1.0)
            path_ang.append(theta)
            found = True
            break

        if not found:
            break

    return path_pts, path_w, path_ang


def _find_endpoints(skel):
    """Pixels du squelette avec exactement 1 voisin (endpoints) ou 3+ (branches)."""
    skel_u8 = (skel > 0).astype(np.uint8)
    kernel   = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skel_u8.astype(np.float32), -1,
                                   kernel.astype(np.float32)) - skel_u8.astype(np.float32)
    endpoints  = np.argwhere((skel_u8 > 0) & (neighbor_count == 1))  # (y,x)
    branches   = np.argwhere((skel_u8 > 0) & (neighbor_count >= 3))
    return endpoints, branches


class RoadTracker:
    CATEGORY = "road/centerline"
    FUNCTION = "run"
    RETURN_TYPES  = (ROAD_PATHS, "INT", "IMAGE")
    RETURN_NAMES  = ("road_paths", "n_paths", "debug_image")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "centerline_mask": ("MASK",),
                "confidence_map":  ("MASK",),
                "width_map":       ("MASK",),
            },
            "optional": {
                "background_image": ("IMAGE",),
                "step_size": ("INT", {
                    "default": 12, "min": 3, "max": 40,
                    "tooltip": "Pas S du tracker (pixels). Papier : S=15."}),
                "max_angle_deg": ("INT", {
                    "default": 10, "min": 1, "max": 30,
                    "tooltip": "Angle max de changement de direction par pas. Papier : ±10°."}),
                "max_steps": ("INT", {
                    "default": 800, "min": 50, "max": 3000,
                    "tooltip": "Nombre max de pas par chemin."}),
                "min_path_length": ("INT", {
                    "default": 3, "min": 2, "max": 50,
                    "tooltip": "Longueur minimale d'un chemin (en nombre de points) pour être conservé."}),
            }
        }

    def run(self, centerline_mask, confidence_map, width_map,
            background_image=None, step_size=12, max_angle_deg=10,
            max_steps=800, min_path_length=3):

        def to_np(t):
            m = t[0] if t.dim() == 3 else t
            return m.cpu().numpy().astype(np.float32)

        conf_np    = to_np(confidence_map)
        center_np  = (to_np(centerline_mask) > 0.5).astype(np.uint8)
        width_np   = to_np(width_map)
        H, W = conf_np.shape

        # ── Squelettiser le mask NMS pour avoir des lignes d'1px ──────────────
        skel = skeletonize(center_np > 0).astype(np.uint8)

        # ── Seeds : endpoints d'abord, puis pixels non-visités ────────────────
        endpoints, branches = _find_endpoints(skel)
        visited = set()
        # Marquer les branches comme visités pour éviter de commencer dedans
        for (by, bx) in branches:
            visited.add((int(by), int(bx)))

        all_paths = []

        def _add_path(pts, ws, angs):
            if len(pts) < min_path_length:
                return
            all_paths.append({
                "points": np.array(pts,  dtype=np.float32),   # (N,2) xy
                "widths": np.array(ws,   dtype=np.float32),   # (N,)
                "angles": np.array(angs, dtype=np.float32),   # (N,)
            })

        # Tracker depuis chaque endpoint
        for (ey, ex) in endpoints:
            ey, ex = int(ey), int(ex)
            if (ey, ex) in visited:
                continue
            theta = _initial_direction(ey, ex, conf_np, H, W)
            pts, ws, angs = _track_one(ey, ex, theta, skel, conf_np, width_np,
                                        visited, H, W, step_size, max_angle_deg, max_steps)
            _add_path(pts, ws, angs)

        # Récupérer les pixels du squelette pas encore visités (ex: boucles)
        remaining = np.argwhere((skel > 0) & ~np.isin(
            np.arange(H * W).reshape(H, W),
            [y * W + x for (y, x) in visited] if visited else []
        ))
        # Version simple : itérer sur tous les pixels skel non-visités
        all_skel_pts = np.argwhere(skel > 0)
        for (ry, rx) in all_skel_pts:
            ry, rx = int(ry), int(rx)
            if (ry, rx) in visited:
                continue
            theta = _initial_direction(ry, rx, conf_np, H, W)
            pts, ws, angs = _track_one(ry, rx, theta, skel, conf_np, width_np,
                                        visited, H, W, step_size, max_angle_deg, max_steps)
            _add_path(pts, ws, angs)

        # ── Debug image ───────────────────────────────────────────────────────
        if background_image is not None:
            bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), dtype=np.uint8)

        # Squelette en gris
        debug[skel > 0] = [80, 80, 80]

        colors_bgr = [
            (0, 200, 255), (0, 255, 100), (255, 100, 0),
            (200, 0, 255), (255, 200, 0), (0, 100, 255),
            (100, 255, 0), (255, 0, 100),
        ]
        for i, path in enumerate(all_paths):
            col = colors_bgr[i % len(colors_bgr)]
            pts_i = path["points"].astype(np.int32)
            if len(pts_i) >= 2:
                for a, b in zip(pts_i[:-1], pts_i[1:]):
                    cv2.line(debug, (a[0], a[1]), (b[0], b[1]), col, 2, cv2.LINE_AA)
            cv2.circle(debug, (pts_i[0, 0], pts_i[0, 1]), 5, (0, 255, 255), -1)
            cv2.circle(debug, (pts_i[-1, 0], pts_i[-1, 1]), 5, (0, 180, 255), -1)

        cv2.putText(debug, f"{len(all_paths)} paths tracked",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        debug_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        debug_t   = torch.from_numpy(debug_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return (all_paths, len(all_paths), debug_t)


NODE_CLASS_MAPPINGS        = {"RoadTracker": RoadTracker}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadTracker": "Road Tracker"}
