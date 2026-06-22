#!/bin/bash
# ============================================================
# setup.sh — Installation complète ComfyUI + Tank Pipeline
# Usage : bash setup.sh
# ============================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFY_DIR="$REPO_DIR/ComfyUI"

echo "=========================================="
echo " Tank Pipeline — Setup"
echo "=========================================="
echo ""

# ── 1. Cloner ComfyUI ───────────────────────────────────────
if [ ! -d "$COMFY_DIR" ]; then
  echo "[1/6] Clonage de ComfyUI..."
  git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
else
  echo "[1/6] ComfyUI déjà présent — mise à jour..."
  git -C "$COMFY_DIR" pull
fi

# ── 2. Installer les dépendances ComfyUI ───────────────────
echo ""
echo "[2/6] Installation des dépendances ComfyUI..."
pip install -r "$COMFY_DIR/requirements.txt"

# ── 3. Copier les custom nodes ──────────────────────────────
echo ""
echo "[3/6] Copie des custom nodes..."
for pkg in "$REPO_DIR/custom_nodes"/*/; do
  name=$(basename "$pkg")
  dest="$COMFY_DIR/custom_nodes/$name"
  if [ -d "$dest" ]; then
    echo "  MAJ: $name"
    rsync -a --exclude='__pycache__' --exclude='*.pyc' "$pkg" "$dest/"
  else
    echo "  INSTALL: $name"
    cp -r "$pkg" "$dest"
  fi
done

# ── 4. Installer les dépendances des custom nodes ───────────
echo ""
echo "[4/6] Installation des dépendances des custom nodes..."

pip install opencv-python numpy torch torchvision torchaudio \
            scipy matplotlib Pillow huggingface_hub timm einops \
            transformers wandb xformers 2>/dev/null || true

# requirements.txt de chaque node
for pkg in "$COMFY_DIR/custom_nodes"/*/requirements.txt; do
  if [ -f "$pkg" ]; then
    dir=$(dirname "$pkg")
    name=$(basename "$dir")
    echo "  pip install -r $name/requirements.txt"
    pip install -r "$pkg" --quiet 2>/dev/null || true
  fi
done

# ── 5. Installer UniDepth ───────────────────────────────────
echo ""
echo "[5/6] Installation de UniDepth..."
UNIDEPTH_NODE="$COMFY_DIR/custom_nodes/comfyui_unidepth"
if [ -f "$UNIDEPTH_NODE/UniDepth-main.zip" ]; then
  cd "$UNIDEPTH_NODE"
  unzip -q UniDepth-main.zip -d /tmp/unidepth_install/
  pip install -e /tmp/unidepth_install/UniDepth-main/ --no-deps --quiet
  echo "  UniDepth installé depuis zip local"
else
  echo "  Installation de UniDepth depuis GitHub..."
  pip install git+https://github.com/lpiccinelli-eth/UniDepth.git --no-deps --quiet 2>/dev/null || \
  echo "  ATTENTION: UniDepth non installé — copier UniDepth-main.zip dans custom_nodes/comfyui_unidepth/"
fi
cd "$REPO_DIR"

# ── 6. Copier le workflow ────────────────────────────────────
echo ""
echo "[6/6] Copie des workflows..."
mkdir -p "$COMFY_DIR/user/default/workflows"
cp "$REPO_DIR/workflows/"*.json "$COMFY_DIR/user/default/workflows/" 2>/dev/null || true

# ── Résumé ───────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Installation terminée !"
echo "=========================================="
echo ""
echo "PROCHAINES ETAPES :"
echo "  1. Installer Blender : https://www.blender.org/download/"
echo "  2. Télécharger le modèle T-14 (.glb) et mettre son chemin"
echo "     dans le node BlenderPerspectiveRender > model_3d_path"
echo "  3. Mettre le chemin de Blender dans blender_path"
echo "     (Windows: C:\Program Files\Blender Foundation\Blender 4.x\blender.exe)"
echo "  4. Lancer ComfyUI :"
echo "     cd ComfyUI && python main.py --listen"
echo "  5. Charger workflows/ULTIMATE_PIPE_v2.json dans ComfyUI"
echo ""
echo "Les modèles (DepthAnything, SAM3.1) seront téléchargés"
echo "automatiquement au premier lancement."
