"""AMD ROCm patches pour Hunyuan3D 2.1.

Helpers extraits de gradio_app.py pour le pipeline shape lifecycle sur AMD/ROCm.
Le pattern principal : Shape generated -> unloaded BEFORE subprocess texture ->
lazy reloaded au prochain click qui en a besoin.

Pourquoi : sur 20 GB VRAM AMD, on ne peut pas avoir Shape (~10 GB) + Texture (~11-21 GB
selon num_view) en VRAM en meme temps. Le subprocess texture (run_texture.py) garantit
un cleanup VRAM etanche via os._exit(0).
"""

import gc
import os
import threading

import torch


_generation_lock = threading.Lock()
"""Serialize les clicks gradio (no .queue() dans l'app)."""

TEXTURE_VENV_PYTHON = os.environ.get(
    "HY3D_TEXTURE_VENV_PYTHON",
    os.path.expanduser("~/hy3d-tex-venv/bin/python"),
)
"""Path to the texture venv's Python 3.11 interpreter (must have bpy installed).

Override via env var HY3D_TEXTURE_VENV_PYTHON before launching gradio. The default
points to ~/hy3d-tex-venv/bin/python — adjust to your texture venv location.
Hardcoded (rather than sys.executable) to prevent accidental launch from the shape
venv (Py 3.10), which does not have bpy."""


def _load_shape_worker(args):
    """Charge le shape pipeline depuis le cache HF disk.

    ~30s premier coup, <10s ensuite (cache disk chaud).

    Pourquoi pas .to('cpu') au lieu de del+reload : recopiait 8 GB VRAM dans RAM CPU,
    et le subprocess texture chargeant aussi sa RAM, le total saturait la RAM systeme
    (30 GB) => OOM killer kill gradio.
    """
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline as _ShapePipeline
    worker = _ShapePipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=False,
        device=args.device,
    )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        worker.enable_flashvdm(mc_algo=mc_algo)
    if args.mc_algo and args.mc_algo != 'mc':
        try:
            from hy3dshape.models.autoencoders import SurfaceExtractors as _SurfaceExtractors
            worker.vae.surface_extractor = _SurfaceExtractors[args.mc_algo]()
            print(f"[surface_extractor] forced to '{args.mc_algo}' on VAE")
        except Exception as _e:
            print(f"[surface_extractor] failed to set '{args.mc_algo}': {_e}")
    if args.compile:
        worker.compile()
    return worker


def _ensure_shape_loaded(worker, args):
    """Reload le shape pipeline si worker is None (release apres subprocess texture).

    Returns worker (nouveau ou existant). Le caller doit assigner le retour au
    global i23d_worker pour que la prochaine gen voit le worker charge.
    """
    if worker is None:
        print("[AMD-VRAM] lazy reload shape pipeline from disk cache (~30s)...")
        worker = _load_shape_worker(args)
    return worker


def _release_torch_memory():
    """Libere la VRAM GPU + force glibc a rendre les pages au kernel.

    A appeler APRES que toutes les references au modele soient detruites
    (sinon refcount > 0 -> gc.collect() inefficace). Pattern caller correct :

        old_worker = i23d_worker
        i23d_worker = None      # global cleared AVANT cleanup
        del old_worker          # derniere ref locale detruite
        _release_torch_memory() # refcount=0, cleanup effectif

    Le malloc_trim force glibc a retourner les pages free au kernel, sinon RSS
    reste haut et le subprocess texture peut declencher OOM killer kernel
    (gradio 27 GB RSS au pic observe en v3 sans malloc_trim).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import ctypes as _ctypes
        _ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as _e:
        print(f"[AMD-VRAM] malloc_trim skipped: {_e}")
