"""
Road Mask Complete
==================
Prend un mask fragmenté (route avec trous) et génère la continuité.

Algorithme :
  1. Extraction de toutes les composantes connexes du mask.
  2. Pour chaque composante : calcul de la direction principale (PCA)
     et des deux extrémités (points les plus éloignés dans la direction principale).
  3. Pour chaque paire de composantes : si elles sont proches et alignées,
     on comble le trou en dessinant un segment de largeur inférée.
  4. On retourne le mask complété + une image de debug.
"""

import numpy as np
import torch
import cv2


def _component_info(mask_u8: np.ndarray, label: int, stats, centroids):
    """
    Retourne pour une composante :
      - direction : vecteur normalisé (principal axe PCA)
      - pt_a, pt_b : les deux extrémités dans la direction principale
      - width_est : largeur estimée (axe secondaire PCA)
      - centroid : (x, y)
    """
    x, y, w, h, area = stats[label]
    cx, cy = centroids[label]

    # Pixels de la composante
    component = (mask_u8 == label)
    ys, xs = np.where(component)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)

    if len(pts) < 4:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    direction = Vt[0]                   # axe long
    perp      = Vt[1]                   # axe court
    width_est = 4.0 * s[1] / max(len(pts) ** 0.5, 1.0)  # estimation largeur

    # Projeter sur l'axe long pour trouver les extrémités
    proj = centered @ direction
    idx_a = int(np.argmin(proj))
    idx_b = int(np.argmax(proj))
    pt_a = pts[idx_a]
    pt_b = pts[idx_b]

    # Largeur estimée depuis les pixels perpendiculaires
    proj_perp = centered @ perp
    width_px = max(3.0, float(proj_perp.max() - proj_perp.min()))

    return {
        "direction": direction,
        "pt_a": pt_a,
        "pt_b": pt_b,
        "width_px": width_px,
        "centroid": mean,
        "area": float(len(pts)),
    }


def _angle_between(d1, d2):
    """Angle en degrés entre deux directions (non-orienté, 0-90)."""
    cos = abs(float(np.dot(d1, d2)))
    cos = min(1.0, cos)
    return float(np.degrees(np.arccos(cos)))


def _endpoint_gap(info_a, info_b):
    """
    Retourne (gap_dist, end_a, end_b) pour la paire d'extrémités la plus proche
    entre les deux composantes.
    """
    candidates = [
        (np.linalg.norm(info_a["pt_a"] - info_b["pt_a"]), info_a["pt_a"], info_b["pt_a"]),
        (np.linalg.norm(info_a["pt_a"] - info_b["pt_b"]), info_a["pt_a"], info_b["pt_b"]),
        (np.linalg.norm(info_a["pt_b"] - info_b["pt_a"]), info_a["pt_b"], info_b["pt_a"]),
        (np.linalg.norm(info_a["pt_b"] - info_b["pt_b"]), info_a["pt_b"], info_b["pt_b"]),
    ]
    return min(candidates, key=lambda x: x[0])


def _collinearity(info_a, info_b, end_a, end_b):
    """
    Mesure si les deux extrémités sont bien dans l'axe de chacune des routes.
    Retourne l'angle max entre le vecteur de connexion et les directions des composantes.
    """
    conn = end_b - end_a
    norm = np.linalg.norm(conn)
    if norm < 1e-6:
        return 0.0
    conn_dir = conn / norm
    angle_a = _angle_between(info_a["direction"], conn_dir)
    angle_b = _angle_between(info_b["direction"], conn_dir)
    return max(angle_a, angle_b)


def complete_mask(mask_np: np.ndarray,
                  max_gap_px: float,
                  max_angle_deg: float,
                  min_area_px: int,
                  width_scale: float) -> tuple:
    """
    Retourne (completed_mask, debug_canvas).
    """
    H, W = mask_np.shape
    binary = (mask_np > 0.5).astype(np.uint8)

    # Composantes connexes
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)

    # Récupérer les infos de chaque composante (ignorer label 0 = fond)
    infos = {}
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        info = _component_info(labels, lbl, stats, centroids)
        if info is not None:
            infos[lbl] = info

    # Debug canvas
    debug = np.zeros((H, W, 3), dtype=np.uint8)
    debug[binary > 0] = 80  # masque original en gris

    # Mask de sortie = copie du mask original
    completed = binary.copy().astype(np.float32)

    connections = []  # pour affichage

    lbls = list(infos.keys())
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            ia = infos[lbls[i]]
            ib = infos[lbls[j]]

            # Vérif alignement global des deux composantes
            global_angle = _angle_between(ia["direction"], ib["direction"])
            if global_angle > max_angle_deg:
                continue

            # Paire d'extrémités la plus proche
            gap_dist, end_a, end_b = _endpoint_gap(ia, ib)
            if gap_dist > max_gap_px:
                continue

            # Vérif colinéarité (le vecteur de connexion est-il dans l'axe ?)
            col_angle = _collinearity(ia, ib, end_a, end_b)
            if col_angle > max_angle_deg:
                continue

            # Largeur du pont = moyenne des deux composantes × scale
            w_bridge = int(round(((ia["width_px"] + ib["width_px"]) / 2.0) * width_scale))
            w_bridge = max(1, w_bridge)

            # Dessiner le pont dans le mask
            p1 = (int(round(end_a[0])), int(round(end_a[1])))
            p2 = (int(round(end_b[0])), int(round(end_b[1])))
            cv2.line(completed, p1, p2, 1.0, w_bridge)

            connections.append((p1, p2, w_bridge, gap_dist))

    # Debug : masque complété en blanc, composantes originales en gris clair,
    #         ponts en vert, extrémités en rouge
    debug[completed > 0.5] = 160
    debug[binary > 0] = [200, 200, 200]

    for p1, p2, w, gap in connections:
        cv2.line(debug, p1, p2, (80, 220, 80), max(1, w), cv2.LINE_AA)
        cv2.circle(debug, p1, 4, (255,  80,  80), -1)
        cv2.circle(debug, p2, 4, (255,  80,  80), -1)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        cv2.putText(debug, f"{gap:.0f}px", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1, cv2.LINE_AA)

    # Extrémités de chaque composante
    for lbl, info in infos.items():
        for pt in [info["pt_a"], info["pt_b"]]:
            cv2.circle(debug, (int(pt[0]), int(pt[1])), 5, (255, 180, 0), 1, cv2.LINE_AA)

    n_conn = len(connections)
    cv2.putText(debug,
                f"{len(infos)} composantes | {n_conn} ponts | gap<{max_gap_px}px angle<{max_angle_deg}deg",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    return completed, debug


class RoadMaskComplete:
    """
    Comble les trous dans un mask de route fragmenté.

    Connecte les composantes proches et alignées en dessinant des segments
    de largeur inférée entre leurs extrémités.

    Paramètres clés :
      max_gap_px    — distance maximale entre deux extrémités à connecter
      max_angle_deg — tolérance d'alignement (0 = parfaitement parallèle)
      width_scale   — facteur sur la largeur estimée du pont (1.0 = même largeur)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "max_gap_px": ("FLOAT", {
                    "default": 150.0, "min": 1.0, "max": 2000.0, "step": 5.0,
                    "tooltip": "Distance max (px) entre deux extrémités pour les connecter."}),
                "max_angle_deg": ("FLOAT", {
                    "default": 20.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Tolérance d'angle (degrés) entre les directions des composantes "
                               "et le vecteur de connexion. 0 = parfaitement aligné."}),
                "min_component_area_px": ("INT", {
                    "default": 100, "min": 1, "max": 10000,
                    "tooltip": "Aire minimale (px²) pour ignorer les micro-artefacts."}),
                "width_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Facteur sur la largeur du pont dessiné. "
                               "1.0 = même largeur que les composantes voisines."}),
            }
        }

    RETURN_TYPES  = ("MASK", "IMAGE")
    RETURN_NAMES  = ("completed_mask", "debug_image")
    FUNCTION      = "complete"
    CATEGORY      = "road/insertion"

    def complete(self, mask, max_gap_px=150.0, max_angle_deg=20.0,
                 min_component_area_px=100, width_scale=1.0):

        if mask.dim() == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()

        completed_np, debug_np = complete_mask(
            mask_np,
            max_gap_px   = max_gap_px,
            max_angle_deg= max_angle_deg,
            min_area_px  = min_component_area_px,
            width_scale  = width_scale,
        )

        completed_t = torch.from_numpy(completed_np)
        debug_t     = torch.from_numpy(debug_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (completed_t, debug_t)


NODE_CLASS_MAPPINGS        = {"RoadMaskComplete": RoadMaskComplete}
NODE_DISPLAY_NAME_MAPPINGS = {"RoadMaskComplete": "Road Mask Complete"}
