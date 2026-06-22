# delicat_et_subtil — Tank Pipeline ComfyUI

Custom nodes ComfyUI pour insertion de tank T-14 Armata avec orientation physique correcte.

## Installation rapide

```bash
git clone git@github.com:Aureleee/delicat_et_subtil.git
cd delicat_et_subtil

# Mac/Linux
bash setup.sh

# Windows
setup.bat
```

Le script installe automatiquement ComfyUI dans `delicat_et_subtil/ComfyUI/`.

## Custom nodes inclus

| Node | Rôle |
|---|---|
| `ComfyUI-PerspectiveFields` | Estimation gravité/latitude caméra |
| `comfyui_road_insertion` | Détection routes (SAM3 All Roads), segmentation en quads (TEST), bbox tank, composite |
| `comfyui_blender_render` | Rendu Blender depuis ComfyUI (orientation tank alignée sur la route) |
| `comfyui-depthanythingv2` | Depth map |
| `comfyui_unidepth` | UniDepth — depth métrique en mètres |
| `comfyui_essentials` | Utilitaires (PreviewAny, MaskPreview+) |
| `comfyui-rmbg` | SAM3 segmentation route |
| `comfyui_deeplsd_road` | Détection lignes route (DeepLSD) |

## Après installation

1. **Blender** : installer depuis https://www.blender.org/download/
2. **Modèle T-14** : télécharger le `.glb` et mettre le chemin dans `BlenderPerspectiveRender`
3. **Lancer ComfyUI** : `cd ComfyUI && python main.py --listen`
4. **Charger le workflow** : `workflows/ULTIMATE_PIPE_v2.json` (copié automatiquement dans ComfyUI)

## Workflows

Les workflows sont dans le dossier [`workflows/`](workflows/) :

- **`ULTIMATE_PIPE_v2.json`** — pipeline courant : SAM3 All Roads → TEST (RoadQuadSegments)
  → Road Quad Gravity Sampler → Blender Perspective Render. Le tank est orienté
  automatiquement dans l'axe de la route (direction parallèle aux bords).

## Modèles téléchargés automatiquement au premier run

- `depth_anything_v2_vitl_fp32.safetensors` (DepthAnything V2)
- `sam3.1_multiplex_fp16.safetensors` (SAM3 segmentation)
- UniDepth weights (HuggingFace, au premier appel du node)

## UniDepth — installation manuelle requise

```bash
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git --no-deps
pip install wandb timm einops
```
