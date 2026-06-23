"""
RoadInsertionController
=======================
Calcule N points d'insertion équitablement répartis le long de la centerline
des quads d'une route sélectionnée.

Contraintes :
  - Les points doivent être sur le mask de route (individual_masks)
  - Les points doivent être à au moins `min_edge_margin` (fraction 0→0.5) de
    chaque extrémité de la centerline (évite les quads déformés aux bords)

Outputs :
  debug_image      : masked_debug + points bleus
  insertion_points : INSERTION_POINTS (liste de dicts {x, y, quad_entry, ...})
  n_found          : INT — combien de points ont été trouvés
"""

import numpy as np
import cv2
import torch

ROAD_QUAD_PAIRS  = "ROAD_QUAD_PAIRS"
INSERTION_POINTS = "INSERTION_POINTS"


def _quad_is_valid(q):
    for a, b in [(q[0], q[3]), (q[1], q[2])]:
        v = b.astype(np.float64) - a.astype(np.float64)
        if np.hypot(*v) < 5.0:
            return False
    left_dir  = (q[3] - q[0]).astype(np.float64)
    right_dir = (q[2] - q[1]).astype(np.float64)
    for d in (left_dir, right_dir):
        n = np.hypot(*d)
        if n > 1e-6:
            d /= n
    if np.linalg.norm(left_dir) < 1e-6 or np.linalg.norm(right_dir) < 1e-6:
        return False
    return abs(float(np.dot(left_dir / np.linalg.norm(left_dir),
                             right_dir / np.linalg.norm(right_dir)))) > 0.85


def _chain_path(centres):
    pts = np.asarray(centres, dtype=np.float64)
    n = len(pts)
    if n <= 2:
        return sorted(range(n), key=lambda i: pts[i][1])
    start = int(np.argmin(pts[:, 1]))
    remaining = set(range(n))
    remaining.discard(start)
    path = [start]
    while remaining:
        last = pts[path[-1]]
        nxt = min(remaining, key=lambda i: float(np.hypot(*(pts[i] - last))))
        path.append(nxt)
        remaining.discard(nxt)
    return path


class RoadInsertionController:
    CATEGORY = "road/insertion"
    FUNCTION = "compute"
    RETURN_TYPES  = ("IMAGE", INSERTION_POINTS, "INT", "INT")
    RETURN_NAMES  = ("debug_image", "insertion_points", "n_found", "road_index")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quad_pairs":       (ROAD_QUAD_PAIRS,),
                "individual_masks": ("MASK",),
                "background_image": ("IMAGE",),
                "n_insertions": ("INT", {
                    "default": 3, "min": 1, "max": 20,
                    "tooltip": "Nombre de points d'insertion à placer."}),
            },
            "optional": {
                "road_index": ("INT", {
                    "default": 0, "min": -2, "max": 64,
                    "tooltip": "-2 = toutes les routes (n_insertions par route). "
                               "-1 = route principale (plus de quads). >=0 = route précise."}),
                "min_edge_margin": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.4, "step": 0.01,
                    "tooltip": "Fraction de la centerline à exclure en haut et en bas "
                               "(évite les quads de bord déformés)."}),
            }
        }

    def compute(self, quad_pairs, individual_masks, background_image,
                n_insertions=3, road_index=-1, min_edge_margin=0.1):

        # ── Canvas debug ──────────────────────────────────────────────────────
        bg = (background_image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        H, W = bg.shape[:2]
        debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

        # ── Mask de route (individual_masks : MASK = (B, H, W) ou (H, W)) ────
        if individual_masks.dim() == 3:
            road_mask = (individual_masks.sum(0).cpu().numpy() > 0).astype(np.uint8)
        else:
            road_mask = (individual_masks.cpu().numpy() > 0).astype(np.uint8)
        if road_mask.shape != (H, W):
            road_mask = cv2.resize(road_mask.astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # ── Identifier toutes les routes disponibles ──────────────────────────
        all_road_ids = sorted(set(e.get("road_idx", 0) for e in (quad_pairs or [])))

        # road_index=-2 → toutes les routes ; -1 → principale ; >=0 → précise
        if road_index == -2:
            target_roads = all_road_ids
        elif road_index == -1:
            counts = {}
            for e in (quad_pairs or []):
                ri = e.get("road_idx", 0)
                counts[ri] = counts.get(ri, 0) + 1
            target_roads = [max(counts, key=counts.get)] if counts else []
        else:
            target_roads = [road_index]

        target_road = target_roads[0] if target_roads else -1  # pour le label final

        # Filtrer + valider les quads (toutes les routes cibles)
        valid_entries = []
        for e in (quad_pairs or []):
            if e.get("road_idx", 0) not in target_roads:
                continue
            q = np.asarray(e["quad"], dtype=np.float32)
            if q.shape == (4, 2) and _quad_is_valid(q):
                valid_entries.append(e)

        # ── Dessiner les quads ────────────────────────────────────────────────
        for e in valid_entries:
            q = np.asarray(e["quad"], dtype=np.int32)
            color = e.get("color", (200, 120, 50))
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.polylines(debug, [q], isClosed=True, color=bgr, thickness=1, lineType=cv2.LINE_AA)

        if not valid_entries:
            cv2.putText(debug, "Aucun quad valide", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
            out = torch.from_numpy(cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
                                   .astype(np.float32) / 255.0).unsqueeze(0)
            return (out, [], 0)

        # ── Grouper les quads par route ───────────────────────────────────────
        by_road = {}
        for e in valid_entries:
            ri = e.get("road_idx", 0)
            by_road.setdefault(ri, []).append(e)

        insertion_points = []

        def _find_quad_for_point(px, py, entries):
            """Retourne le quad qui contient (px,py), ou None."""
            for e in entries:
                q = np.asarray(e["quad"], dtype=np.float32)
                if cv2.pointPolygonTest(q.reshape(-1, 1, 2), (float(px), float(py)), False) >= 0:
                    return e
            return None

        def _distribute_on_road(entries, n):
            """
            1. Construit la centerline à partir des milieux des bords haut/bas des quads.
            2. Distribue n points régulièrement le long de cette centerline.
            3. Pour chaque point :
               a. Vérifie qu'il est sur le mask de route.
               b. Cherche dans quel quad il se trouve.
               c. Si hors-quad (ou hors-mask), déplace le point par spirale croissante
                  jusqu'à trouver une position valide (sur mask ET dans un quad).
            """
            # Centerline = suite ordonnée des milieux des bords haut et bas
            mids = []
            for e in entries:
                q = np.asarray(e["quad"], dtype=np.float64)
                mids.append((q[0] + q[1]) * 0.5)
                mids.append((q[3] + q[2]) * 0.5)
            order    = _chain_path(mids)
            path_pts = [mids[i] for i in order]
            pts_arr  = np.array(path_pts)

            # Dessiner la centerline en blanc
            for a, b in zip(path_pts[:-1], path_pts[1:]):
                cv2.line(debug, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (255, 255, 255), 1, cv2.LINE_AA)

            diffs   = np.diff(pts_arr, axis=0)
            seglens = np.hypot(diffs[:, 0], diffs[:, 1])
            cum     = np.concatenate([[0.0], np.cumsum(seglens)])
            total   = float(cum[-1])

            lo = min_edge_margin
            hi = 1.0 - min_edge_margin
            ts = [0.5] if n == 1 else list(np.linspace(lo, hi, n))

            pts_out = []
            for t_frac in ts:
                t_dist = float(np.clip(t_frac, 0.0, 1.0)) * total
                k = int(np.searchsorted(cum, t_dist, side="right") - 1)
                k = min(max(k, 0), len(pts_arr) - 2)
                seg_len = float(cum[k + 1] - cum[k])
                frac    = 0.0 if seg_len < 1e-9 else (t_dist - cum[k]) / seg_len
                pt      = pts_arr[k] * (1 - frac) + pts_arr[k + 1] * frac
                px = int(np.clip(int(round(pt[0])), 0, W - 1))
                py = int(np.clip(int(round(pt[1])), 0, H - 1))

                # Chercher une position valide : sur le mask ET dans un quad
                best_e = _find_quad_for_point(px, py, entries)
                if best_e is None or road_mask[py, px] == 0:
                    found = False
                    for r in range(1, 60):
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                if abs(dx) != r and abs(dy) != r:
                                    continue
                                nx = int(np.clip(px + dx, 0, W - 1))
                                ny = int(np.clip(py + dy, 0, H - 1))
                                if road_mask[ny, nx] == 0:
                                    continue
                                e = _find_quad_for_point(nx, ny, entries)
                                if e is not None:
                                    px, py, best_e = nx, ny, e
                                    found = True; break
                            if found: break
                        if found: break
                    if not found:
                        continue  # aucune position valide trouvée, on saute ce point

                pts_out.append({"x": px, "y": py,
                                "quad_entry": best_e,
                                "road_idx": best_e.get("road_idx", 0)})

                cv2.circle(debug, (px, py), 10, (220, 80, 0), 2, cv2.LINE_AA)
                cv2.circle(debug, (px, py), 4,  (255, 180, 0), -1, cv2.LINE_AA)
                label = f"r{best_e.get('road_idx',0)} ({px},{py})"
                cv2.putText(debug, label, (px + 12, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 100), 1, cv2.LINE_AA)
            return pts_out

        for ri, entries in sorted(by_road.items()):
            insertion_points.extend(_distribute_on_road(entries, n_insertions))

        roads_label = ",".join(str(r) for r in sorted(by_road.keys()))
        cv2.putText(debug,
                    f"roads={roads_label}  n={len(insertion_points)}",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200),
                    1, cv2.LINE_AA)

        out_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        out_t = torch.from_numpy(out_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (out_t, insertion_points, len(insertion_points), int(target_road))


NODE_CLASS_MAPPINGS = {
    "RoadInsertionController": RoadInsertionController,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RoadInsertionController": "Road Insertion Controller",
}
