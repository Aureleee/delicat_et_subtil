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


# ─── Registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "PerspectiveFieldsLoader":    PerspectiveFieldsLoader,
    "PerspectiveFieldsInference": PerspectiveFieldsInference,
    "RoadGravitySampler":         RoadGravitySampler,
    "PerspectiveFields3DViewer":  PerspectiveFields3DViewer,
    "RoadGravityOffsetEstimator": RoadGravityOffsetEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerspectiveFieldsLoader":    "Perspective Fields Loader V2",
    "PerspectiveFieldsInference": "Perspective Fields Inference V2",
    "RoadGravitySampler":         "Road Gravity Sampler V2",
    "PerspectiveFields3DViewer":  "Perspective Fields 3D Viewer V2",
    "RoadGravityOffsetEstimator": "Road Gravity Offset Estimator",
}
