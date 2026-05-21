# Hunyuan3D-2.1 — AMD ROCm Low-VRAM Fork

> **Unofficial** AMD ROCm fork with low-VRAM patches.
> Original project: [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).
> I am not the upstream author or maintainer of Hunyuan3D-2.1, its weights, or its architecture. This fork only documents how I made it run on a single 20 GB AMD GPU.
>
> This is my first OSS contribution. The optimization ideas and the porting/testing strategy are mine — I empirically tested each VRAM-related knob; the implementation was assisted by [Claude Code](https://www.anthropic.com/claude-code) (Anthropic's CLI) and reviewed by Codex. PRs and issues welcome.

## What this fork is

A working port of **Hunyuan3D-2.1 (shape + texture, PBR, 6-view default)** on **AMD ROCm Linux**, observed at **~10.5 GB VRAM peak** instead of the commonly reported 21–30 GB.

**Key framing — no quality loss, just time adaptation.** Every optimization in this fork is mathematically identical to the upstream pipeline (modulo ~1e-5 numerical drift on the chunked SDPA). The dial you turn for low-VRAM machines is **attention chunk size**: smaller chunks mean less peak memory and slightly more wall-clock time per attention step. The architecture, weights, hyperparameters, and output quality are untouched.

**Who benefits.** AMD ROCm Linux is the primary target (I built this on RX 7900 XT 20 GB). NVIDIA users likely benefit too: most optimizations are cross-platform (DINO offload, VAE slicing, text_encoder offload, Diffusers cache pitfall, subprocess pattern, upstream bug fixes) — see [PORTING_GUIDE.md](PORTING_GUIDE.md) for the per-patch classification. On a low-VRAM NVIDIA card (e.g. 8 GB RTX 4060) the same chunked-SDPA dial should let you fit the 6-view PBR pipeline.

Tested setup:

| Component | Value |
| --- | --- |
| GPU | AMD Radeon RX 7900 XT (gfx1100, **20 GB VRAM**) |
| CPU | AMD Ryzen 9 7900X |
| OS | Ubuntu 22.04.5 LTS, kernel 6.8 |
| ROCm | 6.1.0 system + PyTorch wheels built against ROCm 6.3 |
| PyTorch | `2.7.1+rocm6.3` |
| Python (shape) | 3.10.12 |
| Python (texture) | 3.11.15 (deadsnakes — needed for `bpy==4.5.0`) |

> If you are on **NVIDIA / Windows**, this fork is not your one-click solution, but the underlying VRAM tricks still apply. See [PORTING_GUIDE.md](PORTING_GUIDE.md) for an LLM-friendly adaptation guide.

## TL;DR — the tweaks & tricks

If you just want the short list of what makes this fork different from upstream:

1. **CUDA → ROCm port** — `hipify-perl -inplace` on the `custom_rasterizer` kernel sources, `FORCE_CUDA=1 + CXXFLAGS=-I/opt/rocm/include`, rebuild from source. PyTorch ROCm wheels (`+rocm6.3`). flash-attn from the Triton ROCm fork — **optional**, upstream defaults to `use_flash_attn=False`. One Python 3.11 venv (the unified launcher runs both shape and texture in it; `bpy>=4.x` needs 3.11+). Full recipe in [How I ported from CUDA to ROCm](#how-i-ported-from-cuda-to-rocm-the-meta-trick).
2. **Shape ↔ texture VRAM separation** — subprocess pattern + lazy reload. `os._exit(0)` to bypass the slow C++ leak detector on the child side. `malloc_trim(0)` so RSS comes back down on the parent.
3. **DINO offload** to CPU right after feature extraction (~700 MB peak saved, zero quality cost).
4. **Manual `text_encoder.to("cpu")`** in PBR mode (it's never called there, ~200–500 MB).
5. **`enable_vae_slicing()`** on the multiview pipeline (~0.5–1 GB on final decode).
6. **Chunked SDPA on query** via `HY3D_SDPA_QUERY_CHUNK` — the lever that brings 6-view PBR under 20 GB on AMD. Tunable in both directions: lower for VRAM, higher for speed.
7. **Diffusers `custom_pipeline` cache clear** before `from_pretrained` (otherwise the cache silently ignores patches in the repo — a real hour-eater).
8. **Device-fix in `modules.py`** so the `torch.cat` survives the manual offload above. No value change.
9. **Upstream bugs fixed** — `len(enhance_images["albedo"])` IndexError on `max_num_view<2`; view selection that silently forced 6 views before adaptive selection; `mc_algo` CLI flag not propagated to the VAE surface_extractor (caused a full mesh instead of a hollow one).

Each one is explained in detail in the [Optimizations explained](#optimizations-explained) section below.

## Headline numbers (observed on RX 7900 XT)

> All numbers are **empirical observations on a single Radeon RX 7900 XT (gfx1100, 20 GB VRAM, ROCm 6.3, PyTorch 2.7.1+rocm6.3)**, including realistic background load (browser, desktop). They are not theoretical maxima and will vary on other GPUs, drivers, or PyTorch versions.

| Stage | Mode | VRAM peak | Time |
| --- | --- | --- | --- |
| Shape gen | octree 384 | ~12 GB | ~15 s |
| Shape gen | octree 512 | ~16 GB | ~25 s |
| Texture gen | 1 view (`front_mirror`), low-VRAM optims not yet active | ~12 GB | ~52 s |
| Texture gen | **6 views (default, PBR)**, low-VRAM optims active | **~10.5 GB** | ~129 s |

The 1-view and 6-view numbers are **not directly comparable**: the 1-view row was measured before the ROCm low-VRAM optims (DINO offload + chunked SDPA + device-fix) were applied, while the 6-view row is with those optims active. With the low-VRAM optims, the 1-view peak should drop by a few GB compared to the table value.

Shape and texture run sequentially as separate processes (the texture pipeline is a subprocess of the gradio process) to guarantee a clean VRAM state between them.

## Quick start

This walkthrough uses **one canonical Python 3.11 venv** for both shape and texture. The unified launcher runs the gradio process inside this venv and spawns texture generation as a subprocess in the same venv, so a second 3.10 venv is no longer needed (`bpy >= 4.x` requires 3.11+ anyway).

```bash
git clone -b amd-rocm https://github.com/aminookayamin-gif/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1

# 1) Python 3.11 venv (deadsnakes PPA on Ubuntu 22.04, system Python on Ubuntu 24+)
python3.11 -m venv ~/hy3d-tex-venv
source ~/hy3d-tex-venv/bin/activate

# 2) PyTorch ROCm wheels (adjust the +rocmX.Y suffix to your driver / wheel availability)
pip install torch==2.7.1+rocm6.3 torchvision \
    --index-url https://download.pytorch.org/whl/rocm6.3

# 3) Python deps + setuptools downgrade (pytorch-lightning 1.9.5 still uses pkg_resources)
pip install -r requirements_texture.txt
pip install "setuptools<70"

# 4) ROCm build env vars — needed for any C++/HIP compile step below
export PYTORCH_ROCM_ARCH=gfx1100     # your AMD arch; gfx1100 = RDNA3 RX 7900 series
export ROCM_PATH=/opt/rocm
export FORCE_CUDA=1                  # tells setup.py to build the GPU code path
export CXXFLAGS="-I/opt/rocm/include"

# 5) diso (Differentiable Surface Operators) — required for --mc_algo dmc (hollow meshes).
#    diso ships only CUDA sources upstream; on AMD you need to hipify + build from source.
#    Quick fallback if you want to skip this: launch with --mc_algo mc instead (solid meshes,
#    no diso needed, lower quality on shapes with cavities).
git clone https://github.com/SarahWeiii/diso.git /tmp/diso && pushd /tmp/diso
for f in src/*.cu src/*.cpp src/*.h; do hipify-perl -inplace "$f"; done
pip install .
popd

# 6) custom_rasterizer ROCm module — setup.py auto-detects ROCm PyTorch and picks the
#    HIP sources committed in this fork (see hy3dpaint/custom_rasterizer/setup.py).
pushd hy3dpaint/custom_rasterizer && python setup.py install && popd

# 7) mesh_inpaint_processor (use sysconfig, NOT python3-config: the latter points to the
#    system Python 3.10 on Ubuntu 22.04 and produces a .cpython-310-*.so the 3.11 venv refuses)
pushd hy3dpaint/DifferentiableRenderer
rm -f *.so
EXT_SUFFIX=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
c++ -O3 -Wall -shared -std=c++11 -fPIC \
    $(python -m pybind11 --includes) mesh_inpaint_processor.cpp \
    -o "mesh_inpaint_processor${EXT_SUFFIX}"
popd

# 8) Launch (gradio UI on 127.0.0.1:7860)
bash amd-rocm-extras/launch_hunyuan_unified.sh
```

**`flash-attn` is optional.** Upstream code defaults to `use_flash_attn=False` (see `hy3dshape/hy3dshape/models/denoisers/hunyuandit.py`), so a ROCm flash-attn build is **not** needed unless you opt in. If you do need it, build from the [`ROCm/flash-attention`](https://github.com/ROCm/flash-attention) fork (branch `main_perf`) and set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` in your shell **before** any `import flash_attn`.

**Env vars the launch scripts read** (with defaults — override before launching if your layout differs):

| Var | Default | Used by |
| --- | --- | --- |
| `HY3D_REPO` | `$HOME/Hunyuan3D-2.1` | both `.sh` launchers + `run_texture.py` |
| `HY3D_TEXTURE_VENV` | `$HOME/hy3d-tex-venv` | both `.sh` launchers |
| `HY3D_TEXTURE_VENV_PYTHON` | `$HY3D_TEXTURE_VENV/bin/python` | `amd_rocm_patches.py` |
| `HY3D_RUN_TEXTURE_SCRIPT` | `$HY3D_REPO/amd-rocm-extras/run_texture.py` | `gradio_app.py` |
| `HY3D_SDPA_QUERY_CHUNK` | `256` | `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py` |
| `HF_HOME` | `~/.cache/huggingface` | `run_texture.py` (Diffusers cache clear path) |

## What this fork changes vs upstream

Five stable feature categories on top of `Tencent-Hunyuan/Hunyuan3D-2.1`:

1. **ROCm build support for `custom_rasterizer`** — HIP sources committed alongside the upstream CUDA sources, and `hy3dpaint/custom_rasterizer/setup.py` auto-selects HIP vs CUDA based on `torch.version.hip`. NVIDIA build behavior is preserved unchanged.

2. **Low-view texture selection and PBR baking fixes** — `max_num_view < 6` is honored properly (upstream silently forced 6 views before adaptive selection), `front_mirror` mode for cheap two-view bake, `enhance_images["albedo"]` length fix (upstream IndexError when fewer than 2 views), and `--mc_algo dmc` propagation to the VAE surface_extractor (upstream flag was passed only to FlashVDM).

3. **Low-VRAM texture optimizations** — DINO offload after feature extraction, manual `text_encoder.to("cpu")` in PBR mode, VAE slicing on the multiview pipeline, chunked SDPA on the query dim via `HY3D_SDPA_QUERY_CHUNK`, device safety fix in `modules.py` to survive the manual offload, and a Diffusers `custom_pipeline` cache clear before each `from_pretrained` (otherwise patches in the repo are silently ignored).

4. **Subprocess texture lifecycle and launch helpers** — Shape and texture run sequentially with VRAM release between stages (`malloc_trim`, refcount-safe `gc.collect`, lazy reload). The texture pipeline is spawned as a subprocess and `os._exit(0)`s to bypass the slow C++ leak detector. Driver scripts in `amd-rocm-extras/`: `run_texture.py`, `launch_hunyuan_unified.sh`, `launch_texture.sh`.

5. **AMD ROCm documentation and porting guide** — This file (`README_AMD_ROCM.md`), the LLM-friendly [PORTING_GUIDE.md](PORTING_GUIDE.md), `amd-rocm-extras/README.md`, a scoped MIT license for the original additions, and a banner in the root `README.md`.

For the exact change set, use `git log <upstream-tip>..amd-rocm` and `git diff <upstream-tip>..amd-rocm`.

## Optimizations explained

### 1. DINO offload after feature extraction
[`hy3dpaint/utils/multiview_utils.py`](hy3dpaint/utils/multiview_utils.py)

`dinov2-giant` (~1 GB fp16) is used **once** to extract image features, then sits in VRAM doing nothing during the entire multi-step denoise. We move it to CPU as soon as features are extracted, while moving the features themselves onto the pipeline execution device.

- **Cost:** zero, math identical.
- **Gain:** ~700 MB peak observed.

### 2. Chunked SDPA on query
[`hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py`](hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py)

`attn_refview` runs cross-view attention on a 4D tensor `(batch, heads, seq, dim)`. On ROCm 6.3, **neither the FLASH nor the EFFICIENT SDPA kernel handles this shape** — only the Math kernel does, and it materializes the full QK^T matrix (1.41–5.62 GB transient peak).

We chunk the **query** dimension while keeping K/V full. Cross-view coordination is preserved (every query token still attends to every key token across views). Activated by env var `HY3D_SDPA_QUERY_CHUNK=256`.

- **Cost:** softmax recomputed per chunk → ~1e-5 numerical drift, undetectable visually.
- **Gain:** brings the 6-view peak under 20 GB.

**Tuning `HY3D_SDPA_QUERY_CHUNK`** — the math is identical at any chunk size, only the transient VRAM peak of the attention computation changes. You can move the dial in either direction:

| `HY3D_SDPA_QUERY_CHUNK` | Effect |
| --- | --- |
| unset / `0` | Chunking disabled, fallback to a single `scaled_dot_product_attention` call (= upstream behavior, highest peak). |
| `32`, `64`, `128` | **Lower VRAM peak, slower attention.** Try this if `256` still OOMs on your card (e.g. 8 GB NVIDIA). |
| `256` (baseline) | Observed safe default on my 20 GB AMD setup. |
| `384`, `512`, `768`, `1024` | **Faster attention, higher peak.** Try this if you have VRAM headroom and want to recover speed. |
| `>= query length` | Equivalent to unset (the patch only chunks when `query.shape[-2] > chunk_size`). The actual query length depends on `--resolution` and `max_num_view`; on 6-view default at 512 res it sits in the 1k–2k range. |

Watch `radeontop` / `nvidia-smi` during the denoise step (the long progress bar) — the peak you observe there is what scales with this knob.

### 3. VAE slicing
[`amd-rocm-extras/run_texture.py`](amd-rocm-extras/run_texture.py)

`enable_vae_slicing()` on the multiview pipeline. The final VAE decode loops one sample at a time instead of processing the 12 samples (6 albedo + 6 metallic-roughness) in parallel.

- **Cost:** small wall-clock overhead.
- **Gain:** ~0.5–1 GB on the decode peak.

### 4. Manual text_encoder offload (PBR mode)
[`amd-rocm-extras/run_texture.py`](amd-rocm-extras/run_texture.py)

In PBR mode, the custom pipeline uses learned `learned_text_clip_*` embeddings stored inside the UNet (see `pipeline.py:268-276`). `encode_prompt()` is **never called** — but the text encoder is still loaded to VRAM.

We `text_encoder.to("cpu")` right after the pipeline is built, guarded by `pbr_setting=True` so a non-PBR fallback would still work.

- **Cost:** zero in PBR mode.
- **Gain:** ~200–500 MB depending on fragmentation.

### 5. Device fix in `modules.py`
[`hy3dpaint/hunyuanpaintpbr/unet/modules.py`](hy3dpaint/hunyuanpaintpbr/unet/modules.py)

Once you manually offload `text_encoder` to CPU, the Diffusers device map gets confused and may put the latent `sample` on CPU while the embedding tensors are on CUDA, triggering `RuntimeError: Expected all tensors to be on the same device` right before the `torch.cat`.

We force all three tensors (`sample`, `embeds_normal`, `embeds_position`) onto `next(self.unet.parameters()).device` before the cat. **No value change**, only a safety belt for the offload.

### 6. Diffusers custom_pipeline cache clear
[`amd-rocm-extras/run_texture.py`](amd-rocm-extras/run_texture.py)

`DiffusionPipeline.from_pretrained(custom_pipeline=...)` copies the custom pipeline source into `~/.cache/huggingface/modules/diffusers_modules/local/` on first load, then **reuses the cache without checking the hash of the source repo**. Patching `hy3dpaint/hunyuanpaintpbr/*.py` in the repo does nothing — the cache silently runs the old version.

We `shutil.rmtree` the cache before every `from_pretrained`. Negligible cost (re-copies a few .py files). This pitfall costs hours when you don't know about it.

### 7. Subprocess pattern + `malloc_trim`
[`amd_rocm_patches.py`](amd_rocm_patches.py)

Shape (~10 GB) and Texture (~10–11 GB) cannot coexist on a 20 GB card. We:

1. Run shape in the gradio process.
2. **Release** shape (`global = None; del; gc.collect(); empty_cache(); malloc_trim(0)`) — refcount-safe ordering, otherwise gc keeps it alive.
3. Spawn texture as a **subprocess** that owns its own VRAM lifecycle. It calls `os._exit(0)` at the end to bypass the slow C++ leak detector of `bpy`/`Hunyuan3DPaintPipeline` (otherwise the parent gradio process waits 30–120 s for child cleanup).
4. **Lazy reload** shape on the next click that needs it.

`malloc_trim(0)` from glibc forces the allocator to return freed pages back to the kernel — without it, gradio RSS stays at ~27 GB and the subprocess texture can trigger the OOM killer even though VRAM is fine.

## How I ported from CUDA to ROCm (the meta trick)

This is the bird's-eye view of the porting pattern. The same recipe also worked for `diso` (the differentiable surface library used by Hunyuan3D-2.1's `mc_algo dmc`) — that port lives in a separate repo.

### The 5-step recipe

1. **`hipify-perl -inplace` the CUDA kernel sources.** ROCm ships `hipify-perl` (in `/opt/rocm/bin/`). It does a textual rewrite of `cuda*` APIs → `hip*` APIs:
   ```bash
   cd hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/
   for f in *.cu *.cpp *.h; do hipify-perl -inplace "$f"; done
   for f in *.cu; do mv "$f" "${f%.cu}.hip"; done
   ```
   For most CUDA libraries this is 80–90% of the work. The remaining 10–20% is manual fixes for things hipify doesn't translate well (rare — none needed here).

2. **Set the build env vars.**
   ```bash
   export PYTORCH_ROCM_ARCH=gfx1100        # your AMD arch (gfx1100 = RDNA3 7900 series)
   export ROCM_PATH=/opt/rocm
   export FORCE_CUDA=1                     # build the GPU branch of setup.py
   export CXXFLAGS="-I/opt/rocm/include"   # so HIP headers resolve at compile time
   ```
   `FORCE_CUDA=1` is misleading — it just tells `setup.py` "build the GPU code path even if no CUDA toolkit is detected". The actual compiler is `hipcc` because of the `PYTORCH_ROCM_ARCH` hint.

3. **Compile from source, no wheels.** `python setup.py install` from inside the texture venv (Python 3.11). The Quick Start above does this once because the unified launcher uses 3.11 for both shape and texture; if you maintain a second venv with a different Python version (e.g. an old 3.10 shape venv), you would need to rebuild the C++/HIP modules there separately — the `.so` files carry the Python ABI suffix and are not interchangeable across versions.

4. **PyTorch ROCm wheels.**
   ```bash
   pip install torch==2.7.1+rocm6.3 torchvision \
       --index-url https://download.pytorch.org/whl/rocm6.3
   ```
   `torch.cuda.is_available()` returns `True` (PyTorch ROCm aliases the CUDA API). `torch.version.hip` returns the actual HIP version.

5. **flash-attn via the Triton ROCm fork.** The PyPI `flash-attn` is CUDA-only. Build from [`ROCm/flash-attention`](https://github.com/ROCm/flash-attention) branch `main_perf`. At runtime, set `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` **before** `import flash_attn`, otherwise it tries to load the CUDA backend and crashes with `ImportError`.

### The Python-version trap (specific to this project)

The texture pipeline imports `bpy` (Blender as a Python module). `bpy >= 4.x` requires **Python 3.11+**. Ubuntu 22.04 ships Python 3.10, so you must install Python 3.11 yourself (via the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa)) before building any venv. On Ubuntu 24+ or Windows where Python 3.11+ is the system default, this step is automatic.

In addition, the texture venv needs `pip install "setuptools<70"` because `pytorch-lightning==1.9.5` still imports `pkg_resources`, which `setuptools 70+` removed.

### Where this generalizes

Any CUDA project whose GPU code is bog-standard CUDA (no warp-shuffle hardcodes, no cuDNN / cuBLAS / NVTX, no vendor-specific intrinsics) usually ports in a single afternoon with this recipe. The `diso` port took ~30 minutes once I had the pattern down. Projects that use warp shuffles with hardcoded sizes (32 on NVIDIA, 64 on AMD CDNA / 32 on RDNA) or vendor libraries will need manual fixes — none of that was needed here.

## What I tried that didn't work

These optimizations look obvious and **do not work** with this specific custom pipeline. Saving you the time of trying them:

| Optimization | Why it fails |
| --- | --- |
| `enable_attention_slicing("auto")` | `SlicedAttnProcessor` expects a 3D tensor `(batch, seq, dim)`. The custom UNet manipulates 4D `(batch, heads, seq, dim)` via `AttnCore.process_attention_base`. Silent swap, then `ValueError: too many values to unpack (expected 3)` on forward. |
| `sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION])` (ROCm 6.3) | `torch.backends.cuda.has_flash_sdp` returns `True`, but no kernel handles the 4D `attn_refview` shape → `RuntimeError: No available kernel. Aborting execution.` Math fallback works but is the source of the OOM peak. |
| `enable_model_cpu_offload()` | Installs accelerate hooks on submodules. The custom pipeline does direct calls like `self.vae.encode(images)` that **bypass the hooks** → VAE stays on CPU, input is on CUDA → `RuntimeError: Expected all tensors to be on the same device`. The custom pipeline does not honor the hookable contract. |

If you find a way to make any of these work, PRs welcome.

## Files modified

| File | Reason |
| --- | --- |
| `gradio_app.py` | Shape lifecycle, subprocess pattern, `mc_algo dmc` propagation to VAE surface_extractor, `hy3dpaint` import guard for Py 3.10 (no bpy). |
| `amd_rocm_patches.py` | New module — shape lifecycle helpers. |
| `hy3dpaint/utils/multiview_utils.py` | DINO offload after feature extraction. |
| `hy3dpaint/utils/pipeline_utils.py` | View selection respects `max_num_view<6` (front-locked when =1). |
| `hy3dpaint/textureGenPipeline.py` | `mirror_back` config + fix `len(enhance_images)` IndexError. |
| `hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py` | Chunked SDPA on the query dim. |
| `hy3dpaint/hunyuanpaintpbr/unet/modules.py` | Manual offload safety fix in the cross-condition cat. |
| `hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/*_hip.*` | HIP sources from `hipify-perl` port. Required to rebuild the ROCm custom_rasterizer. |
| `requirements_amd.txt` | Pinned deps for shape venv. |
| `requirements_texture.txt` | Pinned deps for texture venv (numpy<2 for bpy, no cupy, no bpy in shape venv). |
| `amd-rocm-extras/run_texture.py` | Texture subprocess runner. |
| `amd-rocm-extras/launch_*.sh` | Bash launchers with ROCm env vars + `HY3D_SDPA_QUERY_CHUNK`. |

## One venv vs two (historical note)

The Quick Start above uses **one canonical Python 3.11 venv** because the unified launcher runs gradio (shape pipeline) inside the 3.11 venv and spawns texture as a subprocess in the same venv. This is the simplest setup and what I recommend.

During development I temporarily used a two-venv setup (Python 3.10 for shape, Python 3.11 for texture) before realizing the 3.11 venv handles both. If you ever want to split them again — for example to keep a clean shape-only 3.10 venv for the upstream `pip install` workflow — note that:

- `compile_mesh_painter.sh` uses `python3-config`, which on Ubuntu 22.04 points to the **system** Python 3.10. Building in a 3.11 venv with this script produces a `.cpython-310-*.so` extension the 3.11 venv refuses to import. Fix: use `python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"` from inside the venv instead (the Quick Start already does this).
- Every C++/HIP extension (`custom_rasterizer`, `mesh_inpaint_processor`, optionally `flash-attn`, `diso`) has to be rebuilt in each venv — the ABI suffix differs and the `.so` files are not interchangeable.

## Gotchas (specific to AMD ROCm)

- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` must be set **before** `import flash_attn`. Otherwise it tries to load the CUDA module that does not exist on ROCm → ImportError.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` — ROCm 6.3 logs `not supported on this platform` and silently ignores it. Kept for the day a future ROCm version supports it.
- `HIP_VISIBLE_DEVICES=0` is mandatory if your CPU exposes an integrated GPU (Ryzen 7900X has one at bus 0e, separate from the RX 7900 XT at bus 03). Without it, PyTorch may pick the iGPU.
- AMD GPU-accelerated browser tabs (YouTube, etc.) hold 1–3 GB and fragment VRAM. **Close them before running texture**, otherwise the OOM killer may fire even if `radeontop` shows enough free.
- "Hang" perception: between diffusion → Real-ESRGAN → bake → inpaint there are 30–60 s windows with no GPU activity and no Python output. Wait 2–3 minutes before judging a real hang.

## Adapting to NVIDIA / Windows

See [PORTING_GUIDE.md](PORTING_GUIDE.md). It is structured so you can paste it into an LLM (Claude, ChatGPT, etc.) together with your `nvidia-smi` / `pip freeze` output and get usable adaptation steps.

## Search discoverability

This fork is intended to be discoverable for queries like:

- "Hunyuan3D ROCm low VRAM"
- "AMD Hunyuan3D"
- "Hunyuan3D 20GB VRAM"
- "PBR texture low VRAM"
- "Hunyuan3D AMD RX 7900 XT"
- "Hunyuan3D hipify-perl port"

If you arrived here via a different search and the documentation didn't answer your question, please open an issue with the query you used — that's good signal to improve the docs.

**Suggested GitHub topics** (add manually via GitHub UI → repo home → ⚙️ next to "About" → Topics):

```
hunyuan3d
hunyuan3d-2-1
amd
rocm
low-vram
3d-generation
texture-generation
pbr
diffusers
rx-7900-xt
```

## Credits

- **Hunyuan3D-2.1** — Tencent Hunyuan team. Model architecture, weights, training, custom pipeline. [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1).
- **diso AMD ROCm port** — adapted in-place from [SarahWeiii/diso](https://github.com/SarahWeiii/diso) via `hipify-perl` (not part of this fork — separate repo).
- **flash-attn Triton ROCm** — [ROCm/flash-attention](https://github.com/ROCm/flash-attention) `main_perf` branch.
- **Low-VRAM patches and ROCm port** in this fork — Amin ([@aminookayamin-gif](https://github.com/aminookayamin-gif)), with code review by Codex.

### About this fork

I am not the upstream author or maintainer of Hunyuan3D-2.1. This is my first public open-source contribution.

The optimization ideas, porting strategy, and empirical VRAM testing are mine: I identified which parts of the pipeline were holding memory, tested the knobs, and validated the final 20 GB AMD setup. The implementation was assisted by Claude Code, and the patch set was reviewed with Codex.

If something can be improved, issues and PRs are welcome.

## License

This fork combines two licenses. Read both before redistributing.

**Tencent Hunyuan 3D 2.1 Community License Agreement** (upstream):

- The original Hunyuan3D-2.1 source code, weights, configuration, assets, and architecture remain Tencent's property under the upstream license (see [`LICENSE`](LICENSE) and [`Notice.txt`](Notice.txt)).
- This fork **does not** change that license, **does not** redistribute the model weights, and **does not** relicense any upstream file.
- Tencent-derived files that I patched (`gradio_app.py`, the in-tree `hy3dpaint/...` patches, `hy3dpaint/custom_rasterizer/setup.py`) remain subject to the Tencent Community License as combined works. The HIP sources under `hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel/*_hip.*` are derivative works of the Tencent CUDA sources (mechanical translation via `hipify-perl`) and inherit the same upstream licensing.

**MIT License** (my standalone additions and patch portions):

- My wholly original files — `amd-rocm-extras/*`, `amd_rocm_patches.py`, `README_AMD_ROCM.md`, `PORTING_GUIDE.md`, `requirements_amd.txt`, `requirements_texture.txt` — are released under MIT.
- The patch chunks I introduced inside Tencent-derived files are also released under MIT, **but only my patch portions** — not the surrounding upstream code. The combined files remain subject to the Tencent Community License.
- Full scoping is documented in [`amd-rocm-extras/LICENSE`](amd-rocm-extras/LICENSE).
