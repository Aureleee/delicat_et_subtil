import numpy as np
import torch
import cv2
import math

from perspective2d import PerspectiveFields
from perspective2d.perspectivefields import model_zoo
from perspective2d.utils import draw_perspective_fields


# ─── helpers ──────────────────────────────────────────────────────────────────

def comfy_image_to_bgr(tensor):
    img = tensor[0].cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def np_rgb_to_comfy(img_rgb):
    t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


# ─── Node 1 : Model Loader ────────────────────────────────────────────────────

class PerspectiveFieldsLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (list(model_zoo.keys()), {
                    "default": "Paramnet-360Cities-edina-centered",
                }),
            }
        }

    RETURN_TYPES = ("PERSPECTIVE_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "PerspectiveFields"

    def load_model(self, model_version):
        print(f"[PerspectiveFields] Loading: {model_version}")
        model = PerspectiveFields(version=model_version).eval()
        print(f"[PerspectiveFields] Model ready.")
        return (model,)


# ─── Node 2 : Inference ───────────────────────────────────────────────────────

class PerspectiveFieldsInference:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("PERSPECTIVE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "GRAVITY_FIELD", "LATITUDE_FIELD")
    RETURN_NAMES = ("vector_image", "gravity_field", "latitude_field")
    FUNCTION = "run"
    CATEGORY = "PerspectiveFields"

    def run(self, image, model):
        img_bgr = comfy_image_to_bgr(image)
        img_rgb = img_bgr[..., ::-1].copy()

        with torch.no_grad():
            pred = model.inference(img_bgr)

        up   = pred["pred_gravity_original"].cpu()
        lati = pred["pred_latitude_original"].cpu()

        vis_rgb = draw_perspective_fields(
            img_rgb, up, torch.deg2rad(lati), color=(0, 1, 0), return_img=True
        )
        return (np_rgb_to_comfy(vis_rgb), up, lati)


# ─── Node 3 : Road Gravity Sampler ───────────────────────────────────────────

class RoadGravitySampler:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "border_lines":   ("LINE_SEGMENTS",),
                "gravity_field":  ("GRAVITY_FIELD",),
                "latitude_field": ("LATITUDE_FIELD",),
                "road_position": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                }),
                "arrow_scale": ("FLOAT", {
                    "default": 80.0, "min": 10.0, "max": 300.0, "step": 5.0,
                }),
            },
            "optional": {
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("GRAVITY_FIELD", "GRAVITY_FIELD", "GRAVITY_FIELD", "FLOAT", "INT", "INT", "IMAGE")
    RETURN_NAMES  = ("up_vector_3d", "gravity_vector_2d", "road_vector_2d", "latitude_deg", "point_x", "point_y", "debug_image")
    FUNCTION = "sample"
    CATEGORY = "PerspectiveFields"

    def sample(self, border_lines, gravity_field, latitude_field, road_position,
               arrow_scale, background_image=None):
        lines = np.asarray(border_lines["lines"], np.float32)
        W = border_lines["width"]
        H = border_lines["height"]

        if background_image is not None:
            bg = background_image[0].cpu().numpy()
            bg = (bg * 255).clip(0, 255).astype(np.uint8).copy()
            if bg.shape[0] != H or bg.shape[1] != W:
                bg = cv2.resize(bg, (W, H))
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), np.uint8)

        if len(lines) == 0:
            out = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
            return (
                torch.zeros(3), torch.zeros(2), torch.zeros(2), 0.0,
                W // 2, H // 2,
                torch.from_numpy(out.astype(np.float32) / 255.0).unsqueeze(0),
            )

        lengths = np.hypot(lines[:, 1, 0] - lines[:, 0, 0],
                           lines[:, 1, 1] - lines[:, 0, 1])
        order = np.argsort(lengths)[::-1]
        seg_a = lines[order[0]]
        seg_b = lines[order[1]] if len(lines) > 1 else seg_a

        cx_a = (seg_a[0, 0] + seg_a[1, 0]) * 0.5
        cx_b = (seg_b[0, 0] + seg_b[1, 0]) * 0.5
        if cx_a <= cx_b:
            left_seg, right_seg = seg_a, seg_b
        else:
            left_seg, right_seg = seg_b, seg_a

        def orient_by_y(seg):
            return seg if seg[0, 1] <= seg[1, 1] else seg[::-1]
        left_seg  = orient_by_y(left_seg)
        right_seg = orient_by_y(right_seg)

        start_mid = (left_seg[0] + right_seg[0]) * 0.5
        end_mid   = (left_seg[1] + right_seg[1]) * 0.5
        pt  = start_mid * (1.0 - road_position) + end_mid * road_position
        px  = int(np.clip(round(pt[0]), 0, W - 1))
        py  = int(np.clip(round(pt[1]), 0, H - 1))

        gf = gravity_field
        if isinstance(gf, torch.Tensor):
            gvec = gf[:, py, px].cpu().float()
        else:
            gvec = torch.tensor(gf[:, py, px], dtype=torch.float32)

        lf = latitude_field
        lati_deg = float(lf[py, px].cpu()) if isinstance(lf, torch.Tensor) else float(lf[py, px])
        lati_rad = np.deg2rad(lati_deg)
        cos_l = float(np.cos(lati_rad))
        sin_l = float(np.sin(lati_rad))
        ux, uy = float(gvec[0]), float(gvec[1])
        up3d = torch.tensor([ux * cos_l, uy * cos_l, sin_l], dtype=torch.float32)

        def seg_dir(s):
            d = s[1] - s[0]; n = np.hypot(*d)
            return d / n if n > 1e-6 else d

        da = seg_dir(left_seg)
        db = seg_dir(right_seg)
        if np.dot(da, db) < 0:
            db = -db
        road_dir = (da + db) * 0.5
        n = np.hypot(*road_dir)
        if n > 1e-6:
            road_dir /= n
        rvec = torch.tensor(road_dir, dtype=torch.float32)

        for seg in lines:
            p0 = tuple(np.round(seg[0]).astype(int))
            p1 = tuple(np.round(seg[1]).astype(int))
            cv2.line(debug, p0, p1, (80, 80, 80), 2, cv2.LINE_AA)

        def draw_arrow(img, origin, vec, color, scale):
            ox, oy = int(origin[0]), int(origin[1])
            ex = int(ox + vec[0] * scale)
            ey = int(oy + vec[1] * scale)
            cv2.arrowedLine(img, (ox, oy), (ex, ey), color, 3,
                            cv2.LINE_AA, tipLength=0.25)

        draw_arrow(debug, pt, gvec.numpy(), (0, 220, 70),  arrow_scale * abs(cos_l))
        draw_arrow(debug, pt, rvec.numpy(), (255, 140, 0), arrow_scale)

        depth_r = max(3, int(abs(sin_l) * arrow_scale * 0.4))
        cv2.circle(debug, (px, py), depth_r, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(debug, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug, f"up (lat={lati_deg:.1f}deg)", (12, 28),  font, 0.65, (0, 220, 70),  2, cv2.LINE_AA)
        cv2.putText(debug, "road dir",                    (12, 54),  font, 0.65, (255, 140, 0), 2, cv2.LINE_AA)
        cv2.putText(debug, "depth (sin lat)",             (12, 80),  font, 0.65, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(debug, f"up3D=({up3d[0]:.2f},{up3d[1]:.2f},{up3d[2]:.2f})",
                    (12, 106), font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(debug, f"pt=({px},{py})",
                    (12, 126), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        out_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        out_t   = torch.from_numpy(out_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (up3d, gvec, rvec, lati_deg, px, py, out_t)


# ─── Node 4 : 3D Viewer ──────────────────────────────────────────────────────

class PerspectiveFields3DViewer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "up_vector_3d":   ("GRAVITY_FIELD",),
                "road_vector_2d": ("GRAVITY_FIELD",),
                "latitude_deg":   ("FLOAT",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show"
    CATEGORY = "PerspectiveFields"
    OUTPUT_NODE = True

    def show(self, up_vector_3d, road_vector_2d, latitude_deg):
        def to_list(v):
            if isinstance(v, torch.Tensor): return v.cpu().float().tolist()
            return list(v)
        return {"ui": {
            "up3d":   [to_list(up_vector_3d)],
            "road2d": [to_list(road_vector_2d)],
            "lat":    [float(latitude_deg)],
        }}


# ─── Node 5 : Road Gravity Offset Estimator ──────────────────────────────────

class RoadGravityOffsetEstimator:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "up_vector_3d":      ("GRAVITY_FIELD",),
                "gravity_vector_2d": ("GRAVITY_FIELD",),
                "road_vector_2d":    ("GRAVITY_FIELD",),
                "latitude_deg":      ("FLOAT", {"forceInput": True}),
                "K": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Amplitude de la correction. Le signe est automatique : "
                               "positif si la route part vers la droite, négatif vers la gauche. "
                               "0.0 = pas de correction.",
                }),
            }
        }

    RETURN_TYPES  = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("offset_deg", "roll_angle_deg", "road_angle_deg",
                     "unrolled_road_x", "unrolled_road_y")
    FUNCTION      = "estimate"
    CATEGORY      = "PerspectiveFields"

    def estimate(self, up_vector_3d, gravity_vector_2d, road_vector_2d, latitude_deg, K=0.0):
        def to_np(v):
            if isinstance(v, torch.Tensor): return v.cpu().float().numpy()
            return np.array(v, dtype=np.float32)

        gvec = to_np(gravity_vector_2d).flatten()[:2]
        road = to_np(road_vector_2d).flatten()[:2]
        gx, gy = float(gvec[0]), float(gvec[1])
        rx, ry = float(road[0]), float(road[1])

        phi_rad  = math.atan2(-gx, -gy)
        phi_deg  = math.degrees(phi_rad)
        beta_rad = math.atan2(rx, ry)
        beta_deg = math.degrees(beta_rad)

        road_sign  = -1.0 if rx >= 0 else 1.0
        offset_deg = phi_deg * math.sin(beta_rad) * K * road_sign

        rx_lv = -gy*rx - gx*ry
        ry_lv =  gx*rx - gy*ry
        mag = math.sqrt(rx_lv**2 + ry_lv**2)
        if mag > 1e-9: rx_lv /= mag; ry_lv /= mag

        return (float(offset_deg), float(phi_deg), float(beta_deg),
                float(rx_lv), float(ry_lv))


# ─── Node 6 : Road Quad Gravity Sampler (depuis les quads de TEST) ────────────

# Type partagé avec road_quad_segments (sortie quad_pairs du node TEST).
ROAD_QUAD_PAIRS = "ROAD_QUAD_PAIRS"


class RoadQuadGravitySampler:
    """
    Échantillonne up / road / latitude à un point choisi le long des quads de
    route (sortie 'quad_pairs' du node TEST) pour alimenter Blender Perspective
    Render. Remplace RoadGravitySampler (qui prenait des LINE_SEGMENTS).

    Pour le point choisi :
      1. On trouve le quad concerné.
      2. On RECALE le point sur la CENTERLINE = milieu des deux bords du quad.
      3. road_vector_2d = direction des DEUX bords parallèles du quad
         (= vraie orientation de la route depuis la caméra → pas de correction
         d'offset nécessaire, brancher Blender avec offset_deg=0).
      4. up3d = [ux·cos(lat), uy·cos(lat), sin(lat)] depuis gravity + latitude.

    Sélection du point — deux modes :
      • 'slider'   : quads triés haut/gauche → bas/droite ; p∈[0,1] glisse le
                     long de la séquence (interpolation entre 2 quads).
      • 'point_xy' : utilise (point_x, point_y) ; le quad contenant le point est
                     trouvé et le point est recalé sur sa centerline.

    Signature de sortie identique à RoadGravitySampler (drop-in).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quad_pairs":     (ROAD_QUAD_PAIRS,),
                "gravity_field":  ("GRAVITY_FIELD",),
                "latitude_field": ("LATITUDE_FIELD",),
                "selection_mode": (["slider", "point_xy"], {"default": "slider"}),
                "p": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "Mode slider : position le long de la centerline de "
                               "la route (0 = fond/haut, 1 = près de la caméra/bas). "
                               "Paramétrage par longueur d'arc → glisse à vitesse "
                               "constante, même avec les passes hiérarchiques."}),
                "point_x": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "point_y": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "arrow_scale": ("FLOAT", {
                    "default": 80.0, "min": 10.0, "max": 300.0, "step": 5.0}),
                "invert_direction": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Retourne le road_vector de 180° → change le sens du "
                               "tank (avant/arrière)."}),
            },
            "optional": {
                "road_index": ("INT", {
                    "default": -1, "min": -1, "max": 64,
                    "tooltip": "-1 = route principale (celle avec le plus de quads). "
                               ">=0 = sélectionne une route précise (road_idx)."}),
                "background_image": ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("GRAVITY_FIELD", "GRAVITY_FIELD", "GRAVITY_FIELD", "FLOAT",
                     "INT", "INT", "IMAGE")
    RETURN_NAMES  = ("up_vector_3d", "gravity_vector_2d", "road_vector_2d",
                     "latitude_deg", "point_x", "point_y", "debug_image")
    FUNCTION = "sample"
    CATEGORY = "PerspectiveFields"

    @staticmethod
    def _quad_geom(q):
        """Retourne (centre_top, centre_bot, centre, road_dir) pour un quad.

        Ordre des coins (depuis TEST) : [left_top, right_top, right_bot, left_bot]
          bord gauche  = q[0]→q[3]
          bord droit   = q[1]→q[2]
        road_dir pointe haut→bas (vers la caméra), comme RoadGravitySampler.
        """
        ctop = (q[0] + q[1]) * 0.5
        cbot = (q[3] + q[2]) * 0.5
        centre = (ctop + cbot) * 0.5
        dl = q[3] - q[0]
        dr = q[2] - q[1]
        nl, nr = np.hypot(*dl), np.hypot(*dr)
        if nl > 1e-6: dl = dl / nl
        if nr > 1e-6: dr = dr / nr
        if np.dot(dl, dr) < 0: dr = -dr
        d = dl + dr
        nd = np.hypot(*d)
        if nd > 1e-6: d = d / nd
        return ctop, cbot, centre, d

    @staticmethod
    def _side_dirs(q):
        """Directions unitaires des deux bords latéraux (gauche, droite),
        chacune orientée haut→bas (+y). None si le bord est trop court.
          bord gauche = q[0]→q[3], bord droit = q[1]→q[2]
        """
        out = []
        for a, b in [(q[0], q[3]), (q[1], q[2])]:
            v = (b - a).astype(np.float64)
            n = np.hypot(*v)
            if n < 5.0:
                out.append(None)
            else:
                v = v / n
                if v[1] < 0:
                    v = -v
                out.append(v)
        return out[0], out[1]

    @classmethod
    def _is_valid_quad(cls, q):
        """Un quad est valide si ses DEUX bords latéraux existent ET sont
        ~parallèles entre eux (vrai segment de route). Les triangles
        d'extrémité des passes hiérarchiques (un côté dégénéré) sont rejetés.
        """
        dl, dr = cls._side_dirs(q)
        if dl is None or dr is None:
            return False
        return abs(float(np.dot(dl, dr))) > 0.85

    @classmethod
    def _road_dir_from_sides(cls, q):
        """road_vector = droite parallèle aux deux bords latéraux du quad.
        Moyenne des deux directions de bord. Orientée haut→bas (+y).
        """
        dl, dr = cls._side_dirs(q)
        cand = [d for d in (dl, dr) if d is not None]
        if not cand:
            return np.array([0.0, 1.0])
        if len(cand) == 2 and np.dot(cand[0], cand[1]) < 0:
            cand[1] = -cand[1]
        d = cand[0] + (cand[1] if len(cand) == 2 else 0.0)
        n = np.hypot(*d)
        d = d / n if n > 1e-6 else np.array([0.0, 1.0])
        if d[1] < 0:
            d = -d
        return d

    @staticmethod
    def _chain_path(centres):
        """Ordonne les centres en un chemin cohérent via plus-proche-voisin,
        en partant du point le plus haut (plus petit y = fond de la route).
        Robuste aux passes hiérarchiques qui sortent les quads dans le désordre.
        Retourne la liste d'indices ordonnée.
        """
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

    def sample(self, quad_pairs, gravity_field, latitude_field, selection_mode,
               p, point_x, point_y, arrow_scale, invert_direction=False,
               road_index=-1, background_image=None):

        gf, lf = gravity_field, latitude_field
        if isinstance(gf, torch.Tensor):
            H, W = int(gf.shape[-2]), int(gf.shape[-1])
        else:
            gf = np.asarray(gf); H, W = int(gf.shape[-2]), int(gf.shape[-1])

        # ── Collecte / filtrage des quads ─────────────────────────────────────
        # road_index = -1 → route principale (celle avec le plus de quads).
        target_road = road_index
        if target_road < 0:
            counts = {}
            for e in (quad_pairs or []):
                ri = e.get("road_idx", 0)
                counts[ri] = counts.get(ri, 0) + 1
            if counts:
                target_road = max(counts, key=counts.get)

        quads = []
        for e in (quad_pairs or []):
            if target_road >= 0 and e.get("road_idx", 0) != target_road:
                continue
            q = np.asarray(e["quad"], dtype=np.float64)
            if q.shape == (4, 2) and self._is_valid_quad(q):
                quads.append((q, e.get("color", (255, 140, 0))))

        # ── Canvas debug ──────────────────────────────────────────────────────
        if background_image is not None:
            bg = (background_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).copy()
            if bg.shape[0] != H or bg.shape[1] != W:
                bg = cv2.resize(bg, (W, H))
            debug = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        else:
            debug = np.zeros((H, W, 3), np.uint8)

        if not quads:
            out = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
            return (torch.zeros(3), torch.zeros(2), torch.zeros(2), 0.0,
                    W // 2, H // 2,
                    torch.from_numpy(out.astype(np.float32) / 255.0).unsqueeze(0))

        geoms = [self._quad_geom(q) for q, _ in quads]

        # ── Sélection du point ────────────────────────────────────────────────
        if selection_mode == "point_xy":
            P = np.array([float(point_x), float(point_y)])
            best_i = -1
            for i, (q, _) in enumerate(quads):
                if cv2.pointPolygonTest(q.astype(np.float32),
                                        (float(P[0]), float(P[1])), False) >= 0:
                    best_i = i
                    break
            if best_i < 0:
                best_i = int(np.argmin([np.hypot(*(g[2] - P)) for g in geoms]))
            ctop, cbot, _, road_dir = geoms[best_i]
            seg = cbot - ctop
            L2 = float(np.dot(seg, seg))
            t = 0.5 if L2 < 1e-9 else float(np.clip(np.dot(P - ctop, seg) / L2, 0.0, 1.0))
            centre_pt = ctop + t * seg
        else:  # slider : glisse le long de la CENTERLINE (milieux ctop→cbot)
            # Points de la centerline = 2 par quad (milieu bord haut, milieu bord
            # bas) → même un seul quad donne un segment parcourable par p.
            mids = []
            for ctop, cbot, _, _ in geoms:
                mids.append(ctop); mids.append(cbot)
            order = self._chain_path(mids)
            pts = np.array([mids[i] for i in order])
            seglen = np.hypot(*np.diff(pts, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seglen)])
            total = float(cum[-1])
            if total < 1e-9:
                centre_pt = pts[0]
            else:
                target = float(np.clip(p, 0.0, 1.0)) * total
                k = int(np.searchsorted(cum, target, side="right") - 1)
                k = min(max(k, 0), len(pts) - 2)
                seg = cum[k + 1] - cum[k]
                frac = 0.0 if seg < 1e-9 else (target - cum[k]) / seg
                centre_pt = pts[k] * (1 - frac) + pts[k + 1] * frac

        # ── road_vector = droite parallèle aux DEUX BORDS du quad choisi ──────
        # Pas de magie : on prend le quad où tombe le point, et la direction
        # parallèle à ses deux côtés latéraux.
        sel_idx = -1
        for i, (q, _) in enumerate(quads):
            if cv2.pointPolygonTest(q.astype(np.float32),
                                    (float(centre_pt[0]), float(centre_pt[1])),
                                    False) >= 0:
                sel_idx = i
                break
        if sel_idx < 0:
            sel_idx = int(np.argmin([np.hypot(*(g[2] - centre_pt)) for g in geoms]))
        road_dir = self._road_dir_from_sides(quads[sel_idx][0])
        if invert_direction:
            road_dir = -np.asarray(road_dir, dtype=np.float64)

        px = int(np.clip(round(float(centre_pt[0])), 0, W - 1))
        py = int(np.clip(round(float(centre_pt[1])), 0, H - 1))

        # ── Échantillonnage gravity + latitude ────────────────────────────────
        if isinstance(gf, torch.Tensor):
            gvec = gf[:, py, px].cpu().float()
        else:
            gvec = torch.tensor(gf[:, py, px], dtype=torch.float32)

        if isinstance(lf, torch.Tensor):
            lati_deg = float(lf[py, px].cpu())
        else:
            lati_deg = float(np.asarray(lf)[py, px])

        lati_rad = np.deg2rad(lati_deg)
        cos_l, sin_l = float(np.cos(lati_rad)), float(np.sin(lati_rad))
        ux, uy = float(gvec[0]), float(gvec[1])
        up3d = torch.tensor([ux * cos_l, uy * cos_l, sin_l], dtype=torch.float32)
        rvec = torch.tensor(np.asarray(road_dir, dtype=np.float32))

        # ── Debug ─────────────────────────────────────────────────────────────
        for q, color in quads:
            bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.line(debug, tuple(q[0].astype(int)), tuple(q[3].astype(int)), bgr, 1, cv2.LINE_AA)
            cv2.line(debug, tuple(q[1].astype(int)), tuple(q[2].astype(int)), bgr, 1, cv2.LINE_AA)

        # Centerline chaînée (le rail du slider) + extrémités p=0 / p=1
        _mids = []
        for ctop, cbot, _, _ in geoms:
            _mids.append(ctop); _mids.append(cbot)
        path_order = self._chain_path(_mids)
        path_pts = [_mids[i] for i in path_order]
        for a, b in zip(path_pts[:-1], path_pts[1:]):
            cv2.line(debug, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (255, 255, 255), 1, cv2.LINE_AA)
        if len(path_pts) >= 2:
            s, e = path_pts[0], path_pts[-1]
            cv2.circle(debug, (int(s[0]), int(s[1])), 6, (120, 120, 120), -1, cv2.LINE_AA)
            cv2.putText(debug, "p=0", (int(s[0]) + 6, int(s[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(debug, "p=1", (int(e[0]) + 6, int(e[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        def draw_arrow(img, origin, vec, col, scale):
            ox, oy = int(origin[0]), int(origin[1])
            ex, ey = int(ox + vec[0] * scale), int(oy + vec[1] * scale)
            cv2.arrowedLine(img, (ox, oy), (ex, ey), col, 3, cv2.LINE_AA, tipLength=0.25)

        draw_arrow(debug, (px, py), gvec.numpy(), (0, 220, 70), arrow_scale * abs(cos_l))
        # Flèche dessinée inversée pour correspondre au sens visuel du tank
        draw_arrow(debug, (px, py), -np.asarray(road_dir), (255, 140, 0), arrow_scale)
        depth_r = max(3, int(abs(sin_l) * arrow_scale * 0.4))
        cv2.circle(debug, (px, py), depth_r, (0, 180, 255), 2, cv2.LINE_AA)
        cv2.circle(debug, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug, f"up (lat={lati_deg:.1f}deg)", (12, 28), font, 0.6, (0, 220, 70), 2, cv2.LINE_AA)
        cv2.putText(debug, "road dir (borders)", (12, 52), font, 0.6, (255, 140, 0), 2, cv2.LINE_AA)
        cv2.putText(debug,
                    f"mode={selection_mode} road={target_road} pt=({px},{py}) quads={len(quads)}",
                    (12, 76), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        out_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        out_t = torch.from_numpy(out_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        return (up3d, gvec, rvec, lati_deg, px, py, out_t)


# ─── Registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PerspectiveFieldsLoader":    PerspectiveFieldsLoader,
    "PerspectiveFieldsInference": PerspectiveFieldsInference,
    "RoadGravitySampler":         RoadGravitySampler,
    "RoadQuadGravitySampler":     RoadQuadGravitySampler,
    "PerspectiveFields3DViewer":  PerspectiveFields3DViewer,
    "RoadGravityOffsetEstimator": RoadGravityOffsetEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerspectiveFieldsLoader":    "Perspective Fields Loader V2",
    "PerspectiveFieldsInference": "Perspective Fields Inference V2",
    "RoadGravitySampler":         "Road Gravity Sampler V2",
    "RoadQuadGravitySampler":     "Road Quad Gravity Sampler (TEST quads)",
    "PerspectiveFields3DViewer":  "Perspective Fields 3D Viewer V2",
    "RoadGravityOffsetEstimator": "Road Gravity Offset Estimator",
}
