"""
TankScaleFromRoad
=================
Calcule le facteur de scale à appliquer au tank rendu pour qu'il soit
cohérent avec la largeur de la route AU POINT où le tank est posé.

Entrées :
  bbox_width / bbox_height  : dimensions du tank rendu (depuis AlphaBBox)
  active_quad               : le quad sélectionné (depuis RoadQuadGravitySampler)
  point_x / point_y         : position du tank sur l'image (depuis RoadQuadGravitySampler)
  K                         : constante  (scale = 1 → tank = K × largeur route)

Formule :
  t          = projection du point sur l'axe du quad (0=haut, 1=bas)
  left_pt    = lerp(quad[0], quad[3], t)
  right_pt   = lerp(quad[1], quad[2], t)
  road_px    = |dot(right_pt − left_pt, normale_route)|
  avg_bbox   = (bbox_width + bbox_height) / 2
  scale      = (K × road_px) / avg_bbox

Outputs :
  scale          : FLOAT
  debug_string   : STRING
  avg_bbox_px    : FLOAT
  road_width_px  : FLOAT
"""

import numpy as np

ROAD_QUAD_PAIRS = "ROAD_QUAD_PAIRS"


def _road_width_at_point(quad_entry, px, py):
    """Mesure la largeur perpendiculaire du quad au point (px, py)."""
    q = np.asarray(quad_entry["quad"], dtype=np.float64)  # [lt, rt, rb, lb]

    # Direction de route = moyenne des deux bords latéraux
    left_dir  = q[3] - q[0]
    right_dir = q[2] - q[1]
    road_dir  = (left_dir + right_dir) / 2.0
    norm = np.linalg.norm(road_dir)
    if norm < 1e-6:
        return 0.0
    road_dir /= norm
    normal = np.array([-road_dir[1], road_dir[0]])  # perpendiculaire

    # Projeter le point sur l'axe haut→bas du quad pour trouver t
    ctop = (q[0] + q[1]) * 0.5
    cbot = (q[3] + q[2]) * 0.5
    seg  = cbot - ctop
    L2   = float(np.dot(seg, seg))
    P    = np.array([float(px), float(py)])
    t    = 0.5 if L2 < 1e-9 else float(np.clip(np.dot(P - ctop, seg) / L2, 0.0, 1.0))

    # Interpoler les bords gauche et droit à ce t
    left_pt  = q[0] + t * (q[3] - q[0])
    right_pt = q[1] + t * (q[2] - q[1])

    width_px = abs(float(np.dot(right_pt - left_pt, normal)))
    return width_px


class TankScaleFromRoad:
    CATEGORY = "blender/utils"
    FUNCTION = "compute"
    RETURN_TYPES  = ("FLOAT", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES  = ("scale", "debug_string", "avg_bbox_px", "road_width_px")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_width":  ("INT",   {"default": 256, "min": 1, "max": 8192}),
                "bbox_height": ("INT",   {"default": 256, "min": 1, "max": 8192}),
                "active_quad": (ROAD_QUAD_PAIRS,),
                "point_x":     ("INT",   {"default": 0, "min": 0, "max": 8192}),
                "point_y":     ("INT",   {"default": 0, "min": 0, "max": 8192}),
                "K": ("FLOAT", {
                    "default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01,
                    "tooltip": "K=1 → tank = largeur route. K=0.6 → tank = 60% de la route."}),
            }
        }

    def compute(self, bbox_width, bbox_height, active_quad, point_x, point_y, K=0.6):
        avg_bbox = (bbox_width + bbox_height) / 2.0

        if not active_quad:
            return (1.0, "active_quad vide — connecter RoadQuadGravitySampler.active_quad",
                    avg_bbox, 0.0)

        road_px = _road_width_at_point(active_quad[0], point_x, point_y)

        if avg_bbox < 1.0:
            return (1.0, "bbox vide", 0.0, road_px)
        if road_px < 1.0:
            return (1.0, "largeur route nulle au point donné", avg_bbox, 0.0)

        scale = (K * road_px) / avg_bbox
        debug = (f"avg_bbox={avg_bbox:.1f}px  road@({point_x},{point_y})={road_px:.1f}px  "
                 f"K={K}  →  scale={scale:.4f}")
        return (scale, debug, avg_bbox, road_px)


NODE_CLASS_MAPPINGS        = {"TankScaleFromRoad": TankScaleFromRoad}
NODE_DISPLAY_NAME_MAPPINGS = {"TankScaleFromRoad": "Tank Scale From Road"}
