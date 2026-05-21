#!/bin/bash
# Launch Hunyuan3D-2.1 texture gen as a standalone CLI (without gradio).
# Uses the texture venv (Python 3.11 with bpy + max_num_view patches).
#
# Override these env vars before sourcing if your layout differs:
#   HY3D_REPO         : path to the cloned Hunyuan3D-2.1 repo  (default: $HOME/Hunyuan3D-2.1)
#   HY3D_TEXTURE_VENV : path to the texture venv (Python 3.11) (default: $HOME/hy3d-tex-venv)
#
# Usage:
#   bash launch_texture.sh [--view-preset front|weighted_back|weighted_sides|default] \
#                          [--max-num-view N] [--resolution 512] \
#                          <input_mesh.glb> <source_image.png> <output_textured.glb>
#
# Example (front-only, VRAM-efficient):
#   bash launch_texture.sh --view-preset front --resolution 512 \
#       /path/to/shape.glb \
#       /path/to/source_image.png \
#       /tmp/textured.glb

set -e

HY3D_REPO="${HY3D_REPO:-$HOME/Hunyuan3D-2.1}"
HY3D_TEXTURE_VENV="${HY3D_TEXTURE_VENV:-$HOME/hy3d-tex-venv}"
HY3D_RUN_TEXTURE_SCRIPT="${HY3D_RUN_TEXTURE_SCRIPT:-$HY3D_REPO/amd-rocm-extras/run_texture.py}"
export HY3D_REPO

# Activate texture venv (Python 3.11)
source "$HY3D_TEXTURE_VENV/bin/activate"

echo "[launch] HY3D_REPO        : $HY3D_REPO"
echo "[launch] HY3D_TEXTURE_VENV: $HY3D_TEXTURE_VENV"
echo "[launch] which python     : $(which python)"
echo "[launch] python --version : $(python --version)"
echo "[launch] venv             : $VIRTUAL_ENV"

# AMD ROCm env (force the discrete GPU, ignore an integrated GPU if present)
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx1100             # adjust to your AMD arch
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_PATH=/opt/rocm

# Flash-attn Triton mode (safe even if hy3dpaint does not import it)
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# Live logs (no Python stdout buffering)
export PYTHONUNBUFFERED=1

# HIP allocator anti-fragmentation (silently ignored on ROCm 6.3; on NVIDIA use
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True instead)
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Chunked SDPA dial — must be set here too, otherwise --view-preset default (6 views)
# can OOM in this standalone launcher even though it fits in the unified launcher.
# Lower (128/64/32) for less VRAM, higher (384/512/1024) for more speed; math identical.
export HY3D_SDPA_QUERY_CHUNK="${HY3D_SDPA_QUERY_CHUNK:-256}"

cd "$HY3D_REPO"

# Run the texture pipeline
python "$HY3D_RUN_TEXTURE_SCRIPT" "$@"
