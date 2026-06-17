@echo off
REM ============================================================
REM setup.bat — Installation complète ComfyUI + Tank Pipeline
REM Usage : double-clic ou "setup.bat" dans cmd
REM ============================================================

set REPO_DIR=%~dp0
set COMFY_DIR=%REPO_DIR%ComfyUI

echo ==========================================
echo  Tank Pipeline — Setup Windows
echo ==========================================
echo.

REM ── 1. Cloner ComfyUI ──────────────────────────────────────
if not exist "%COMFY_DIR%" (
    echo [1/6] Clonage de ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI.git "%COMFY_DIR%"
) else (
    echo [1/6] ComfyUI deja present — mise a jour...
    git -C "%COMFY_DIR%" pull
)

REM ── 2. Dependances ComfyUI ─────────────────────────────────
echo.
echo [2/6] Installation des dependances ComfyUI...
pip install -r "%COMFY_DIR%\requirements.txt"

REM ── 3. Copier les custom nodes ─────────────────────────────
echo.
echo [3/6] Copie des custom nodes...
for /D %%p in ("%REPO_DIR%custom_nodes\*") do (
    echo   %%~nxp
    xcopy /E /I /Y "%%p" "%COMFY_DIR%\custom_nodes\%%~nxp\" >nul
)

REM ── 4. Dependances des custom nodes ────────────────────────
echo.
echo [4/6] Installation des dependances...
pip install opencv-python numpy torch torchvision torchaudio ^
            scipy matplotlib Pillow huggingface_hub timm einops ^
            transformers wandb

for /R "%COMFY_DIR%\custom_nodes" %%f in (requirements.txt) do (
    pip install -r "%%f" --quiet 2>nul
)

REM ── 5. UniDepth ────────────────────────────────────────────
echo.
echo [5/6] UniDepth...
echo   Installer manuellement si besoin :
echo   pip install git+https://github.com/lpiccinelli-eth/UniDepth.git --no-deps

REM ── 6. Workflow ────────────────────────────────────────────
echo.
echo [6/6] Copie du workflow...
if not exist "%COMFY_DIR%\user\default\workflows" mkdir "%COMFY_DIR%\user\default\workflows"
copy "%REPO_DIR%ULTIMATE_PIPE.json" "%COMFY_DIR%\user\default\workflows\" >nul

echo.
echo ==========================================
echo  Installation terminee !
echo ==========================================
echo.
echo PROCHAINES ETAPES :
echo   1. Installer Blender : https://www.blender.org/download/
echo   2. Ouvrir ComfyUI et charger ULTIMATE_PIPE.json
echo   3. Dans BlenderPerspectiveRender, mettre :
echo      blender_path = C:\Program Files\Blender Foundation\Blender 4.x\blender.exe
echo      model_3d_path = chemin vers t-14_armata.glb
echo   4. Lancer : cd ComfyUI ^&^& python main.py --listen
echo.
pause
