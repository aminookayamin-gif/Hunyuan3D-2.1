#!/bin/bash
# Launch Hunyuan3D-2.1 unified (shape + texture via subprocess pattern) on AMD ROCm.
# Uses the texture venv (Python 3.11 with bpy) so the gradio process can run shape
# natively (DMC hollow mesh) and spawn the texture pipeline as a VRAM-clean subprocess.
#
# Override these env vars before sourcing if your layout differs:
#   HY3D_REPO         : path to the cloned Hunyuan3D-2.1 repo  (default: $HOME/Hunyuan3D-2.1)
#   HY3D_TEXTURE_VENV : path to the texture venv (Python 3.11) (default: $HOME/hy3d-tex-venv)

set -e

HY3D_REPO="${HY3D_REPO:-$HOME/Hunyuan3D-2.1}"
HY3D_TEXTURE_VENV="${HY3D_TEXTURE_VENV:-$HOME/hy3d-tex-venv}"

# Export for child processes (gradio_app.py + run_texture.py read these)
export HY3D_REPO
export HY3D_TEXTURE_VENV
export HY3D_TEXTURE_VENV_PYTHON="${HY3D_TEXTURE_VENV_PYTHON:-$HY3D_TEXTURE_VENV/bin/python}"
export HY3D_RUN_TEXTURE_SCRIPT="${HY3D_RUN_TEXTURE_SCRIPT:-$HY3D_REPO/amd-rocm-extras/run_texture.py}"

# Activate texture venv (Python 3.11 with bpy)
source "$HY3D_TEXTURE_VENV/bin/activate"

echo "[launch] HY3D_REPO        : $HY3D_REPO"
echo "[launch] HY3D_TEXTURE_VENV: $HY3D_TEXTURE_VENV"
echo "[launch] which python     : $(which python)"
echo "[launch] python --version : $(python --version)"
echo "[launch] venv             : $VIRTUAL_ENV"

# Force the discrete AMD GPU, ignore an integrated GPU if present
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Flash-attn Triton ROCm
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export GPU_ARCHS=gfx1100             # adjust to your AMD arch (gfx1100 = RDNA3 7900 series)
export PYTORCH_ROCM_ARCH=gfx1100     # idem
export ROCM_PATH=/opt/rocm

# Live Python logs (no stdout buffering)
export PYTHONUNBUFFERED=1

# HIP allocator anti-fragmentation (silently ignored on ROCm 6.3, kept for future versions).
# On NVIDIA, switch to: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Chunked SDPA on the query dim — patched in
#   hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py
# N = chunk of N query tokens, full K/V kept => cross-view coordination preserved.
# Reduces SDPA transient peak (1.41–5.62 GB without chunking => OOM on 20 GB AMD).
# Math identical modulo ~1e-5 numerical drift.
# 256 = observed safe default on a 20 GB AMD setup. Lower (128/64/32) for more
# VRAM headroom (slower attention). Raise (384/512/768/1024) for speed if you have headroom.
export HY3D_SDPA_QUERY_CHUNK=256

cd "$HY3D_REPO"

# Launch gradio:
# - do NOT pass --disable_tex (we want the "Gen Textured Shape" button visible)
# - no --enable_t23d (tries to import hy3dgen, removed in 2.1)
# - --mc_algo dmc + --enable_flashvdm + --low_vram_mode as documented in README
python gradio_app.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --mc_algo dmc \
    --low_vram_mode \
    --enable_flashvdm \
    --port 7860 \
    --host 127.0.0.1
