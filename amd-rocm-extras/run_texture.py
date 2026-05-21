"""
Hunyuan3D-2.1 texture gen CLI runner (AMD ROCm).
Designed for the texture venv (Python 3.11 with bpy installed + max_num_view patched).

Env vars (with sensible defaults):
  HY3D_REPO   : path to the cloned Hunyuan3D-2.1 repo (default: auto-detect from this script's location, two levels up).

Usage:
  python run_texture.py [--view-preset front|weighted_back|weighted_sides|default] \\
                        [--max-num-view N] [--resolution 512] \\
                        <input_mesh.glb> <source_image.png> <output_textured.glb>

Expected log on --view-preset front:
  [view_selection] [{'azim': 0, 'elev': 0, 'weight': 1}]
"""
import argparse
import os
import sys

PRESET_MAP = {
    "front": 1,
    "front_mirror": 1,     # 1 vue diffusion + mirror H sur la cam back avant bake (cf. config.mirror_back)
    "weighted_back": 2,    # front + back (weights 1.0, 0.5)
    "weighted_sides": 4,   # front + back + right + left
    "default": 6,          # upstream behavior, all 6 canonical views
}

parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 Paint pipeline (AMD ROCm shape-only patched)")
parser.add_argument("--view-preset", choices=list(PRESET_MAP.keys()), default="front",
                    help="View selection preset (default: front = max_num_view=1)")
parser.add_argument("--max-num-view", type=int, default=None,
                    help="Override view count (1-6+). Overrides --view-preset if set.")
parser.add_argument("--resolution", type=int, default=512,
                    help="Multiview diffusion resolution (default 512, try 384 if OOM)")
parser.add_argument("input_mesh", help="Input mesh (.glb or .obj — output of shape gen)")
parser.add_argument("input_image", help="Source image (the original 2D image used to generate the shape)")
parser.add_argument("output_mesh", help="Output textured mesh path (.glb recommended; .obj also accepted)")
args = parser.parse_args()

max_num_view = args.max_num_view if args.max_num_view is not None else PRESET_MAP[args.view_preset]

print(f"[runner] python: {sys.executable}")
print(f"[runner] preset={args.view_preset} max_num_view={max_num_view} resolution={args.resolution}")
print(f"[runner] input_mesh  : {args.input_mesh}")
print(f"[runner] input_image : {args.input_image}")
print(f"[runner] output_mesh : {args.output_mesh}")

# Path setup (same as Hunyuan3D-2.1 demo.py upstream)
# Default: auto-detect repo root from this script's location (amd-rocm-extras/ is at repo root).
# Override via HY3D_REPO env var if you placed this script outside the repo.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HUNYUAN_DIR = os.environ.get("HY3D_REPO", os.path.dirname(_THIS_DIR))
sys.path.insert(0, HUNYUAN_DIR)  # for torchvision_fix.py at repo root
sys.path.insert(0, os.path.join(HUNYUAN_DIR, "hy3dshape"))
sys.path.insert(0, os.path.join(HUNYUAN_DIR, "hy3dpaint"))
os.chdir(HUNYUAN_DIR)
print(f"[runner] HY3D_REPO: {HUNYUAN_DIR}")

# torchvision compat fix
try:
    from torchvision_fix import apply_fix
    apply_fix()
except Exception as e:
    print(f"[runner] torchvision_fix skipped: {e}")

import torch
print(f"[runner] torch {torch.__version__} | hip {torch.version.hip} | cuda avail {torch.cuda.is_available()}")
print(f"[runner] device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

print(f"[load] preparing Hunyuan3DPaintConfig(max_num_view={max_num_view}, resolution={args.resolution})...")
conf = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=args.resolution)
conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
conf.mirror_back = (args.view_preset == "front_mirror")
if conf.mirror_back:
    print("[runner] mirror_back=True (front texture sera dupliquee flip H sur la cam back)")

# AMD ROCm low-VRAM: clear the Diffusers custom_pipeline cache
# AVANT load. Sinon le cache local conserve une copie obsolete du custom_pipeline
# (depuis le 1er load) qui ignore les patches du repo. Le vidage force Diffusers
# a re-copier depuis le repo source patche a chaque subprocess. Negligeable cout
# (re-copy de quelques .py).
import shutil as _shutil_cache
# Respect HF_HOME if set (Hugging Face Hub convention); fall back to ~/.cache/huggingface.
_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
_DIFFUSERS_LOCAL_CACHE = os.path.join(_HF_HOME, "modules", "diffusers_modules", "local")
_shutil_cache.rmtree(_DIFFUSERS_LOCAL_CACHE, ignore_errors=True)
print(f"[optim] cleared diffusers local custom_pipeline cache: {_DIFFUSERS_LOCAL_CACHE}")

print("[load] loading Hunyuan3DPaintPipeline (this can take a moment)...")
paint_pipeline = Hunyuan3DPaintPipeline(conf)

# AMD ROCm low-VRAM optims — no expected visual quality loss, math identical:
# - vae_slicing: VAE.decode boucle 1 sample a la fois au lieu des 12 (6 albedo + 6 mr) en
#   parallele. Filet ~0.5-1 GB sur le decode final.
# NOTE: attention_slicing("auto") testee puis retiree — incompatible avec le custom_pipeline
# hunyuanpaintpbr. Le SlicedAttnProcessor diffusers attend du 3D (batch, seq, dim) mais le
# custom UNet manipule du 4D (batch, heads, seq, dim) via AttnCore.process_attention_base.
# L'enable ne plante pas (swap silencieux) mais le forward crash avec ValueError unpack 3.
# Le levier VRAM principal (offload DINO apres extraction features) est applique dans
# hy3dpaint/utils/multiview_utils.py:100.
mvp = paint_pipeline.models["multiview_model"].pipeline
try:
    mvp.enable_vae_slicing()
    print("[optim] vae_slicing enabled (no expected visual quality loss)")
except Exception as e:
    print(f"[optim] vae_slicing skipped: {e}")

# Manual text_encoder offload (PBR mode never uses it):
# Le custom_pipeline hunyuanpaintpbr utilise des embeddings learned_text_clip_{token}
# pre-appris stockes dans le UNet (cf pipeline.py:268-276). En mode PBR, encode_prompt()
# n'est JAMAIS appele, donc text_encoder est charge en VRAM mais inactif. Gain ~200-500 MB
# selon fragmentation. Check pbr_setting au runtime pour eviter un offload si fallback non-PBR.
import torch as _torch
if hasattr(mvp.unet, 'pbr_setting') and mvp.unet.pbr_setting:
    mvp.text_encoder.to("cpu")
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    print("[optim] text_encoder offloaded to CPU (PBR mode, never used)")
else:
    print("[optim] text_encoder kept on GPU (non-PBR mode active)")

# NOTE: enable_model_cpu_offload() tested and dropped — incompatible with the custom_pipeline.
# enable_model_cpu_offload() installe des hooks accelerate sur les sub-modules, mais
# le custom_pipeline hunyuanpaintpbr fait des appels directs (self.vae.encode(images),
# self.unet(...), etc.) qui BYPASS les hooks. Resultat: VAE reste CPU, input tensor CUDA,
# device mismatch RuntimeError au premier appel vae.encode().
# NOTE: sdpa_kernel([FLASH, EFFICIENT]) tested and dropped — ROCm 6.3 exposes
# ces backends mais aucun ne supporte le shape 4D (batch, heads, seq, dim) du attn_refview
# custom du UNet PBR. Resultat: "RuntimeError: No available kernel. Aborting execution."
# Chunked SDPA on the query dim, patched in
# hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py (relative to repo root).
# Active via env var HY3D_SDPA_QUERY_CHUNK=256 (set dans launch_hunyuan_unified.sh).
# Math identique modulo erreurs numeriques 1e-5 (softmax recompute per chunk).
print(f"[gen] generating texture (look for [view_selection] log just below)...")
chunk_env = os.environ.get("HY3D_SDPA_QUERY_CHUNK", "0")
print(f"[optim] HY3D_SDPA_QUERY_CHUNK={chunk_env} (0=disabled, fallback Math)")
# .glb safety wrapper: upstream textureGenPipeline.save_glb=True does
#   convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
# If output_mesh_path is itself .glb, the .replace is a no-op and OBJ text gets written
# into the .glb path before "conversion" — ambiguous. Always feed a .obj path internally,
# then move/rename the produced .glb to whatever extension the user requested.
_user_output = args.output_mesh
_output_root, _output_ext = os.path.splitext(_user_output)
_internal_obj_path = _output_root + ".obj"
result_path = paint_pipeline(
    mesh_path=args.input_mesh,
    image_path=args.input_image,
    output_mesh_path=_internal_obj_path,
)
# After the call, the pipeline has written <_output_root>.obj AND <_output_root>.glb
# (save_glb=True default). If the user asked for .glb, ensure it sits at the requested path.
if _user_output.lower().endswith(".glb"):
    _expected_glb = _output_root + ".glb"
    if os.path.exists(_expected_glb) and os.path.abspath(_expected_glb) != os.path.abspath(_user_output):
        import shutil as _shutil_move
        _shutil_move.move(_expected_glb, _user_output)
    print(f"[done] textured GLB written to: {_user_output}")
else:
    print(f"[done] textured mesh written to: {result_path}")

# AMD ROCm low-VRAM: force immediate exit to bypass the C++ leak detector of
# bpy / Hunyuan3DPaintPipeline, which takes 30-120s+ printing "Freeing memory after the leak detector"
# and blocks subprocess.run() on the gradio side (so the finally + reload + UI update never run).
# The mesh is already written to disk; exit cleanly without waiting for the C++ cleanup.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
