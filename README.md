# delicat_et_subtil — Vehicle Insertion Pipeline (ComfyUI)

Custom nodes ComfyUI pour insertion de véhicule 3D avec orientation physique correcte dans des images aériennes / drone.

## Installation rapide

```bash
git clone git@github.com:Aureleee/delicat_et_subtil.git
cd delicat_et_subtil

# Windows
setup.bat
```

Le script installe automatiquement ComfyUI dans `delicat_et_subtil/ComfyUI/`, copie tous les custom nodes et les workflows.

## Custom nodes inclus

| Package | Rôle |
|---|---|
| `ComfyUI-PerspectiveFields` | Estimation gravité/latitude caméra, alignement route |
| `comfyui_road_insertion` | Détection routes (SAM3 All Roads), segmentation en quads, composite |
| `comfyui_road_centerline` | Extraction centerline, NMS, tracking, tensor voting, connectivity refine |
| `comfyui_blender_render` | Rendu Blender depuis ComfyUI (orientation alignée sur la route) |
| `comfyui_unidepth` | UniDepth — depth métrique en mètres |
| `comfyui-depthanythingv2` | Depth map (DepthAnything V2) |
| `comfyui_essentials` | Utilitaires (PreviewAny, MaskPreview+) |
| `comfyui-rmbg` | SAM3 segmentation route |
| `comfyui_deeplsd_road` | Détection lignes route (DeepLSD) |

## Après installation

1. **Blender** : installer depuis https://www.blender.org/download/
2. **Modèle 3D** : télécharger un `.glb` et renseigner le chemin dans `BlenderPerspectiveRender`
3. **Lancer ComfyUI** : `cd ComfyUI && python main.py --listen`
4. **Charger le workflow** : `workflows/pipeline_main.json`

## Workflows

Les workflows sont dans le dossier [`workflows/`](workflows/) :

- **`pipeline_main.json`** — pipeline courant : SAM3 All Roads → RoadQuadSegments
  → Road Quad Gravity Sampler → Blender Perspective Render.
  Le véhicule est orienté automatiquement dans l'axe de la route.

## Modèles téléchargés automatiquement

- `depth_anything_v2_vitl_fp32.safetensors` (DepthAnything V2)
- `sam3.1_multiplex_fp16.safetensors` (SAM3 segmentation)
- UniDepth weights (HuggingFace, au premier appel du node)

## UniDepth — installation manuelle requise

```bash
pip install git+https://github.com/lpiccinelli-eth/UniDepth.git --no-deps
pip install wandb timm einops
```

## Structure du repo

```
delicat_et_subtil/
├── custom_nodes/          ← source des nodes (editer ici)
│   ├── comfyui_blender_render/
│   ├── comfyui_road_insertion/
│   ├── comfyui_road_centerline/
│   ├── ComfyUI-PerspectiveFields/
│   └── ...
├── workflows/             ← workflows ComfyUI (.json)
├── setup.bat              ← installation Windows
├── setup.sh               ← installation Mac/Linux
└── ComfyUI/               ← généré par setup (gitignored)
```

> **Note** : les modifications de nodes se font dans `custom_nodes/`.
> Le `setup.bat` copie automatiquement vers `ComfyUI/custom_nodes/`.
