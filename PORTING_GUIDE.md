# Porting Guide — adapting this fork to NVIDIA / Windows

> This document is intentionally structured so you can **paste it into an LLM** (Claude, ChatGPT, etc.) together with your environment details to get usable adaptation instructions. See the prompt template at the bottom.

## Reference baseline (what was tested)

This is the exact setup the fork was developed on. Use it as the **left side** of any adaptation.

| Axis | Value |
| --- | --- |
| GPU vendor | AMD |
| GPU | Radeon RX 7900 XT (gfx1100, 20 GB VRAM) |
| OS | Linux (Ubuntu 22.04.5 LTS, kernel 6.8) |
| Backend | ROCm 6.1 system + PyTorch wheels built against ROCm 6.3 |
| PyTorch | `torch==2.7.1+rocm6.3` |
| Python (shape venv) | 3.10.12 |
| Python (texture venv) | 3.11.15 (deadsnakes) |
| `bpy` | 4.5.0 (Python 3.11+ only) |
| custom_rasterizer | rebuilt from HIP sources (in this repo) |
| Result | 6-view PBR texture in ~10.5 GB VRAM peak, 129 s on RX 7900 XT |

## Classification of changes in this fork

Each modification falls into one of three buckets. The bucket determines what happens on a different setup.

### A. Cross-platform — keep as-is

These help anywhere VRAM is tight, regardless of vendor or OS:

| Change | File | Why it helps on NVIDIA / Windows too |
| --- | --- | --- |
| DINO offload to CPU after feature extraction | `hy3dpaint/utils/multiview_utils.py` | DINO sits idle during denoise everywhere. Pure win. |
| Manual `text_encoder.to("cpu")` in PBR mode | `amd-rocm-extras/run_texture.py` | PBR mode never calls `encode_prompt()`. Pure win. |
| `enable_vae_slicing()` | `amd-rocm-extras/run_texture.py` | Diffusers feature, vendor-agnostic. |
| `shutil.rmtree(diffusers_modules/local)` before `from_pretrained` | `amd-rocm-extras/run_texture.py` | Pitfall of `custom_pipeline=`. Affects every setup using a custom_pipeline. |
| Subprocess pattern + `os._exit(0)` for texture | `gradio_app.py`, `amd-rocm-extras/run_texture.py` | Cross-platform technique for hard VRAM separation. |
| Refcount-safe release sequence + `gc.collect()` + `empty_cache()` | `amd_rocm_patches.py` | Pure Python / Diffusers idiom. |
| `mc_algo dmc` propagation to VAE surface_extractor | `gradio_app.py` | Upstream bug, vendor-agnostic. |
| `len(enhance_images["albedo"])` instead of `len(enhance_images)` | `hy3dpaint/textureGenPipeline.py` | Upstream IndexError on `max_num_view<2`. Vendor-agnostic. |
| `max_num_view<6` honored in view selection | `hy3dpaint/utils/pipeline_utils.py` | Upstream forces 6 views before any adaptive selection. Vendor-agnostic. |
| Device-fix in `modules.py` before the `torch.cat` | `hy3dpaint/hunyuanpaintpbr/unet/modules.py` | Only relevant if you manually offload submodules. No value change. |
| `face_reduce_worker` removed pre-texture | `gradio_app.py` | Quality decision (paint on full-res mesh). Vendor-agnostic. |

### B. ROCm-specific — replace or skip on NVIDIA

| Change | File | What to do on NVIDIA |
| --- | --- | --- |
| HIP custom_rasterizer sources | `hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/*_hip.*` | **Auto-handled.** `hy3dpaint/custom_rasterizer/setup.py` now detects `torch.version.hip` and selects either the HIP sources (AMD) or the upstream CUDA sources (NVIDIA). On NVIDIA you don't need to do anything — the upstream `*.cu` / `*.cpp` files are still used. |
| `diso` (DMC backend) | not in this repo — third-party | **Recommended on both sides** if you want hollow-mesh output (`--mc_algo dmc`). On NVIDIA: `pip install diso` (CUDA wheels exist upstream). On AMD: hipify the CUDA sources from [`SarahWeiii/diso`](https://github.com/SarahWeiii/diso) and build from source (see Quick Start in `README_AMD_ROCM.md`). **Fallback:** launch with `--mc_algo mc` to skip diso entirely — produces solid meshes instead of hollow ones (lower quality on shapes with internal cavities). |
| `requirements_amd.txt` | repo root | **Replace** with upstream `requirements.txt` + your CUDA wheel selection. The pins here are ROCm-specific. |
| `requirements_texture.txt` | repo root | Use as a starting point. The numpy<2 / `setuptools<70` constraints come from `bpy==4.5.0` + `pytorch-lightning==1.9.5`, which are vendor-agnostic. Replace the torch line with your CUDA wheel. |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (env) | `amd-rocm-extras/launch_*.sh` | **Remove.** On NVIDIA, flash-attn loads its prebuilt CUDA module automatically. |
| `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `GPU_ARCHS=gfx1100`, `PYTORCH_ROCM_ARCH=gfx1100`, `ROCM_PATH=/opt/rocm`, `PYTORCH_HIP_ALLOC_CONF=...` | `amd-rocm-extras/launch_*.sh` | **Replace** with `CUDA_VISIBLE_DEVICES=0` and (optionally) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` which actually works on CUDA. |
| Chunked SDPA via `HY3D_SDPA_QUERY_CHUNK` | `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py` | **Keep optional.** On NVIDIA, FLASH and EFFICIENT SDPA kernels **usually/likely** work for this 4D shape, so **leave `HY3D_SDPA_QUERY_CHUNK` unset first** (the patch falls back to the original single SDPA call). Only enable it if you still OOM — on a low-VRAM NVIDIA card (e.g. 8 GB) start low (`128`, `64`, `32`). Lower = less peak, slower attention, math identical. Full dial documented in `README_AMD_ROCM.md` section "Chunked SDPA on query". |
| `enable_flashvdm` + `--low_vram_mode` | `gradio_app.py` CLI flags | Both are upstream flags, **keep on**. They are not AMD-specific. |

### C. Linux-specific — replace on Windows

| Change | File | What to do on Windows |
| --- | --- | --- |
| `HY3D_TEXTURE_VENV_PYTHON` env var (defaults to `$HOME/hy3d-tex-venv/bin/python`) | `amd_rocm_patches.py` | Set `HY3D_TEXTURE_VENV_PYTHON=C:\Users\<you>\hy3d-tex-venv\Scripts\python.exe` before launching gradio. |
| `source "$HY3D_TEXTURE_VENV/bin/activate"` in `.sh` launchers | `amd-rocm-extras/launch_*.sh` | Rewrite as `.bat` / `.ps1` using `%HY3D_TEXTURE_VENV%\Scripts\activate.bat`. |
| `libc.so.6` + `malloc_trim(0)` | `amd_rocm_patches.py` | **Wrap in try/except** (it's already wrapped). On Windows the `CDLL` call silently fails and is skipped, no harm. Windows allocator doesn't keep RSS as high anyway. |
| `python3-config` EXT_SUFFIX issue | build instructions for `mesh_inpaint_processor.cpp` | On Windows you build with `cl.exe` / `setup.py build_ext` instead of `c++`. Use the same `sysconfig.get_config_var('EXT_SUFFIX')` trick to get `.pyd` for the right Python version. |
| `bash compile_mesh_painter.sh` | `hy3dpaint/DifferentiableRenderer/` | Translate to `python setup.py build_ext --inplace` or a `.bat` calling MSVC. |

## Adaptation matrices

### AMD ROCm Linux → NVIDIA CUDA Linux

**Easy.** Most of what you do is **subtract** ROCm-specific things:

1. **Don't touch `custom_rasterizer` sources.** `hy3dpaint/custom_rasterizer/setup.py` detects ROCm vs CUDA PyTorch (via `torch.version.hip`) and selects the correct sources automatically — on NVIDIA it uses the upstream `*.cu` / `*.cpp` files unchanged.
2. Replace `requirements_amd.txt` with upstream `requirements.txt` + your CUDA torch wheel (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`).
3. Remove ROCm env vars from `launch_*.sh`. Optionally set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
4. **Leave `HY3D_SDPA_QUERY_CHUNK` unset first** — NVIDIA SDPA kernels usually handle the 4D `attn_refview` shape. Only set it (start low: `128` / `64` / `32`) if you still OOM. See bucket B above for the full dial.
5. Keep all bucket-A changes — they help.

You may not need the subprocess pattern if you have >= 24 GB VRAM and run shape + texture comfortably together. But the pattern is harmless either way.

### AMD ROCm Linux → NVIDIA CUDA Windows

= (Linux → Windows) **then** (AMD → NVIDIA), in that order:

1. First do the Windows path / venv / scripts translation (bucket C).
2. Then do the NVIDIA substitution (bucket B).
3. WSL2 is an alternative that lets you keep the Linux shell scripts unchanged with CUDA passthrough.

### AMD ROCm Linux → AMD ROCm Windows

As of early 2026, **PyTorch on ROCm for Windows is not officially supported**. Options to investigate:

1. **ZLUDA** (CUDA → HIP translation layer) — experimental, coverage of kernels is partial. May fail on the same 4D `attn_refview` shape as native ROCm, but worth trying since the failure mode is well-defined (you can fall back to `HY3D_SDPA_QUERY_CHUNK=128` if FLASH/EFFICIENT kernels are not picked up).
2. **WSL2 + ROCm** — ROCm-on-WSL2 has improved over time but remains less reliable than native Linux. Worth a try if you don't want to dual-boot.
3. **Dual-boot Linux** — the safest path to native ROCm performance.

If you get any of these working, please open an issue with your setup details and observed VRAM/timings.

### NVIDIA Windows → NVIDIA Linux

Out of scope but trivial: pretty much just `wsl --install`. Then follow "NVIDIA CUDA Linux".

## What you should NOT touch

These are upstream architecture / weights, not optimizations:

- Anything under `hy3dshape/` except as upstream evolves
- `hy3dpaint/hunyuanpaintpbr/pipeline.py` (the custom_pipeline itself)
- `hy3dpaint/cfgs/*.yaml`
- Model weights and downloads from Hugging Face (`tencent/Hunyuan3D-2.1`)

If you need to adapt these, you should be talking to the upstream maintainers, not to this fork.

## Verification checklist after porting

When you adapt this fork to your stack, run these smoke tests **in order** — each one isolates a different subsystem so a failure points to the right culprit:

1. **Basic GPU detection.**
   ```bash
   python -c "import torch; print('cuda.is_available =', torch.cuda.is_available()); print('device =', torch.cuda.get_device_name(0)); print('torch.version.hip =', torch.version.hip); print('torch.version.cuda =', torch.version.cuda)"
   ```
   On AMD ROCm: `torch.version.hip` is set, `torch.version.cuda` is `None`. On NVIDIA: the opposite. `cuda.is_available()` returns `True` in both cases (PyTorch ROCm aliases the CUDA API).

2. **Shape pipeline only** — `python gradio_app.py --disable_tex --mc_algo dmc --enable_flashvdm` then click "Gen Shape" once. Confirms: `torch`, `hy3dshape`, FlashVDM, and **`diso`** (only loaded when `--mc_algo dmc`; if `diso` is missing or broken, you'll get `ImportError: Please install diso via pip install diso, or set mc_algo to 'mc'` — fall back to `--mc_algo mc` to keep going). Does **not** exercise `custom_rasterizer` yet (that's bake-time).

3. **Texture pipeline as CLI** — `bash amd-rocm-extras/launch_texture.sh --view-preset front <shape.glb> <image.png> /tmp/textured.glb`. Confirms: `bpy`, `diffusers` with the `custom_pipeline` cache clear, the `hunyuanpaintpbr` custom_pipeline, multiview rendering, **`custom_rasterizer`** (bake), Real-ESRGAN, and the `.glb` safety wrapper.

4. **VRAM scaling under stress** — re-run step 3 with `--view-preset default` (6 views) and watch VRAM (`nvidia-smi -l 1` / `radeontop -b <bus>`). If you OOM, set `HY3D_SDPA_QUERY_CHUNK=128` (or lower: `64`, `32`) and retry. If you still OOM at `32`, the bottleneck is somewhere other than `attn_refview` — check `nvidia-smi` for other processes holding VRAM.

5. **End-to-end through gradio** — launch `amd-rocm-extras/launch_hunyuan_unified.sh`, generate a shape, then click "Gen Textured Shape". Confirms the full shape → subprocess → texture lifecycle including the VRAM release between stages.

## Prompt template — paste this into your LLM

If you do not want to read this whole guide, copy the block below and fill in your machine details. Any modern LLM should produce a working adaptation plan.

````markdown
I want to adapt the following fork to my machine:

https://github.com/aminookayamin-gif/Hunyuan3D-2.1 (branch: amd-rocm)

The author's baseline is:
- GPU: AMD Radeon RX 7900 XT (20 GB, gfx1100, ROCm)
- OS: Linux (Ubuntu 22.04)
- Backend: ROCm 6.1/6.3 + torch==2.7.1+rocm6.3
- Python: one canonical 3.11 venv (bpy 4.5 needs >=3.11; the unified launcher runs both shape and texture in it)
- DMC backend: `diso` (built from source via hipify-perl; falls back to `--mc_algo mc` if absent)

My setup is:
- GPU: <FILL: nvidia-smi -L output, e.g. RTX 4070 12 GB>
- OS: <FILL: Windows 11 / Ubuntu / WSL2 etc>
- Driver: <FILL: nvidia-smi top header>
- CUDA: <FILL: nvcc --version OR torch.version.cuda>
- Python: <FILL: python --version>
- Current pip freeze (relevant parts): <FILL>

Please read the following two documents from the repo:
1. README_AMD_ROCM.md — the headline doc and per-commit summary
2. PORTING_GUIDE.md — the file you are reading right now, which contains the
   classification of changes (cross-platform / ROCm-specific / Linux-specific).

Based on that classification:

1. List the files in the fork that I MUST adapt for my stack, and what to change.
2. List the files I can keep as-is from the fork.
3. List the files I should restore from upstream Tencent-Hunyuan/Hunyuan3D-2.1.
4. Give me an installation script for my venv(s) using the correct torch wheel.
5. Give me a launch script equivalent to amd-rocm-extras/launch_hunyuan_unified.sh
   adapted to my OS and GPU vendor.
6. List the verification steps from PORTING_GUIDE.md adapted to my hardware
   monitoring tools.

Do not hallucinate file paths. If a file path is uncertain, tell me which
command to run to find it.
````

The more detail you give in `<FILL>`, the better the answer. `nvidia-smi -L`, `nvidia-smi`, `nvcc --version`, `python --version`, `pip freeze | head -30` are all useful.

## If you get stuck

- Open an issue on the fork with: your OS, GPU, driver, what you ran, the full error message.
- Mention which bucket (A/B/C) of changes you suspect, and which file. This guide is structured to make that easy.
- If you found a bug specific to the fork (not upstream), I will look at it. If it is an upstream bug, please report it on `Tencent-Hunyuan/Hunyuan3D-2.1` instead.
