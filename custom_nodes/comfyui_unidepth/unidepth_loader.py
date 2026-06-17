"""
UniDepth Loader Node
====================
Télécharge et charge un modèle UniDepth (V1 ou V2) depuis HuggingFace.

Modèles disponibles :
  - unidepth-v2-vits14  (ViT-S  ~22M params)  → rapide, léger
  - unidepth-v2-vitb14  (ViT-B  ~86M params)  → bon équilibre
  - unidepth-v2-vitl14  (ViT-L ~307M params)  → meilleur, recommandé
  - unidepth-v1-vitl14  (ViT-L V1)            → ancienne version
  - unidepth-v1-cnvnxtl (ConvNext-L V1)       → alternative convolutionnelle

Tous entraînés sur un mix de datasets LiDAR/RGBD couvrant scènes intérieures
et extérieures (voir note dans le node).
"""

import torch

# ─── type custom ──────────────────────────────────────────────────────────────
UNIDEPTH_MODEL = "UNIDEPTH_MODEL"

# ─── descriptions des modèles ─────────────────────────────────────────────────
MODEL_INFO = {
    "unidepth-v2-vitl14": {
        "version": "V2", "class": "UniDepthV2",
        "backbone": "ViT-L/14", "params": "~307M",
        "speed": "lent", "quality": "meilleur",
        "recommande": True,
    },
    "unidepth-v2-vitb14": {
        "version": "V2", "class": "UniDepthV2",
        "backbone": "ViT-B/14", "params": "~86M",
        "speed": "moyen", "quality": "bon",
    },
    "unidepth-v2-vits14": {
        "version": "V2", "class": "UniDepthV2",
        "backbone": "ViT-S/14", "params": "~22M",
        "speed": "rapide", "quality": "correct",
    },
    "unidepth-v1-vitl14": {
        "version": "V1", "class": "UniDepthV1",
        "backbone": "ViT-L/14", "params": "~307M",
        "speed": "lent", "quality": "bon (V1)",
    },
    "unidepth-v1-cnvnxtl": {
        "version": "V1", "class": "UniDepthV1",
        "backbone": "ConvNext-L", "params": "~200M",
        "speed": "moyen", "quality": "bon (V1, conv)",
    },
}

MODEL_CHOICES = list(MODEL_INFO.keys())


class UniDepthLoader:
    """
    Charge un modèle UniDepth depuis HuggingFace (téléchargement automatique
    au premier lancement, puis cache local).

    Tous les modèles UniDepth sont entraînés sur un MIX de 9+ datasets :
      - NYUv2         : intérieur RGB-D (Kinect), distances ~0-10m
      - KITTI         : extérieur conduire (LiDAR Velodyne), distances ~0-80m
      - NuScenes      : conduite urbaine multi-caméras (LiDAR), ~0-80m
      - DDAD          : conduite (LiDAR dense), long range
      - ETH3D         : scènes indoor/outdoor, haute précision
      - SUN-RGBD      : intérieur (Kinect/ToF), ~0-10m
      - IBims-1       : benchmark intérieur, annotations soignées
      - Diode (In)    : intérieur/extérieur (LiDAR)
      - ARKitScenes   : scènes iOS LiDAR (V2 only)
      + données propriétaires pour V2

    Différence V1 vs V2 :
      - V2 : confidence map, meilleure netteté des bords (EdgeGuidedLocalSSI),
             plus flexible en résolution, support ONNX, plus rapide
      - V1 : plus simple, pas de confidence

    Pour des scènes de route extérieure : ViT-L V2 recommandé (KITTI, NuScenes, DDAD)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_CHOICES, {
                    "default": "unidepth-v2-vitl14",
                    "tooltip": (
                        "v2-vitl14 : meilleur (lent, ~307M) | "
                        "v2-vitb14 : équilibré (~86M) | "
                        "v2-vits14 : rapide (~22M) | "
                        "v1-vitl14 : ancienne V1 ViT-L | "
                        "v1-cnvnxtl : ancienne V1 ConvNext-L"
                    ),
                }),
                "resolution_level": ("INT", {
                    "default": 5, "min": 0, "max": 9, "step": 1,
                    "tooltip": (
                        "V2 uniquement. Contrôle la résolution interne (0=min, 9=max). "
                        "Plus haut = plus de détails mais plus lent et plus de VRAM. "
                        "5 est un bon défaut."
                    ),
                }),
            }
        }

    RETURN_TYPES  = (UNIDEPTH_MODEL,)
    RETURN_NAMES  = ("unidepth_model",)
    FUNCTION      = "load"
    CATEGORY      = "depth/unidepth"

    def load(self, model_name, resolution_level=5):
        info = MODEL_INFO[model_name]
        hf_repo = f"lpiccinelli/{model_name}"

        if info["version"] == "V2":
            from unidepth.models import UniDepthV2
            model = UniDepthV2.from_pretrained(hf_repo)
            model.resolution_level = resolution_level
            model.interpolation_mode = "bilinear"
        else:
            from unidepth.models import UniDepthV1
            model = UniDepthV1.from_pretrained(hf_repo)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        return ({"model": model, "name": model_name, "info": info},)


NODE_CLASS_MAPPINGS        = {"UniDepthLoader": UniDepthLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"UniDepthLoader": "UniDepth Loader"}
