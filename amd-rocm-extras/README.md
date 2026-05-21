# `amd-rocm-extras/` — scripts and launchers

Three files copied from my actual setup. They are not strictly required to use the fork (you can call `gradio_app.py` directly), but they bundle the ROCm env vars and the subprocess pattern correctly.

## Files

### `run_texture.py`

CLI runner for the texture pipeline. Spawned as a subprocess by `gradio_app.py` after shape generation, so VRAM is fully released between the two stages. Also usable standalone:

```bash
bash launch_texture.sh \
    --view-preset front \
    --resolution 512 \
    /path/to/shape.glb \
    /path/to/source_image.png \
    /path/to/output_textured.glb
```

Available `--view-preset`:

| Preset | `max_num_view` | Notes |
| --- | --- | --- |
| `front` | 1 | Front view only. Fastest, ~52 s on RX 7900 XT. |
| `front_mirror` | 1 | Front view + mirror H flip onto back cam before bake. Cheap "two-view" trick. |
| `weighted_back` | 2 | Front + back, weights `[1.0, 0.5]`. |
| `weighted_sides` | 4 | Front + back + right + left. |
| `default` | 6 | Upstream behavior, full quality PBR. ~10.5 GB VRAM peak on AMD with the low-VRAM patches active. ~129 s on RX 7900 XT. |

Override the preset count: `--max-num-view N`.

Lower diffusion resolution if you OOM: `--resolution 384`.

### `launch_hunyuan_unified.sh`

The main entry point — launches the gradio UI on `127.0.0.1:7860`. Activates the texture venv (Python 3.11 with bpy), sets ROCm env vars, then runs `gradio_app.py` with the correct flags (`--mc_algo dmc`, `--enable_flashvdm`, `--low_vram_mode`). Sets `HY3D_SDPA_QUERY_CHUNK=256` for the chunked SDPA patch in the unet.

> The `256` chunk size is a balanced default. **Lower it** (`128`, `64`, `32`) if you need more VRAM headroom — slower attention, math identical. **Raise it** (`384`, `512`, `768`, `1024`) if you have headroom and want speed back. The full dial is documented in `README_AMD_ROCM.md` section "Chunked SDPA on query".

### `launch_texture.sh`

Wrapper around `run_texture.py` for direct CLI use (without gradio). Same env vars as `launch_hunyuan_unified.sh`.

## Env vars (with sensible defaults)

Both `.sh` scripts and `amd_rocm_patches.py` read these env vars before launching. Defaults are sensible — only override if your layout differs.

| Env var | Default | Used by |
| --- | --- | --- |
| `HY3D_REPO` | `$HOME/Hunyuan3D-2.1` | both `.sh` launchers + `run_texture.py` |
| `HY3D_TEXTURE_VENV` | `$HOME/hy3d-tex-venv` | both `.sh` launchers |
| `HY3D_TEXTURE_VENV_PYTHON` | `$HY3D_TEXTURE_VENV/bin/python` | `amd_rocm_patches.py` (the gradio process spawns the texture subprocess through this interpreter) |
| `HY3D_RUN_TEXTURE_SCRIPT` | `$HY3D_REPO/amd-rocm-extras/run_texture.py` | `gradio_app.py` and `launch_texture.sh` |

Example override (drop these into your `.bashrc` / `.zshrc` if you keep your venvs elsewhere):

```bash
export HY3D_REPO="$HOME/code/Hunyuan3D-2.1"
export HY3D_TEXTURE_VENV="$HOME/.local/share/venvs/hunyuan-tex"
```

You can also place `run_texture.py` outside `amd-rocm-extras/` (for example next to your texture venv) — just point `HY3D_RUN_TEXTURE_SCRIPT` at the new location.

## Notes on the patterns

- `os._exit(0)` at the end of `run_texture.py` is **deliberate** — it bypasses the slow C++ leak detector of `bpy` / `Hunyuan3DPaintPipeline` that otherwise blocks `subprocess.run()` on the parent side for 30–120 s. The mesh is already written to disk by then.
- The diffusers custom_pipeline cache is wiped before every `from_pretrained` (`shutil.rmtree`). This is critical — see the optimization breakdown in `README_AMD_ROCM.md`.
- The text_encoder offload is guarded by `unet.pbr_setting` so a non-PBR fallback would still work.

## License

MIT — see `LICENSE` in this directory. The original Hunyuan3D-2.1 code remains under its own Tencent Community License (see the repo-root `LICENSE` file).
