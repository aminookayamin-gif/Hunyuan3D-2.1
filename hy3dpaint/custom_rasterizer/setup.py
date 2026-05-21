# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from setuptools import setup, find_packages
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# build custom rasterizer
#
# AMD ROCm support (amd-rocm fork): on a ROCm-built PyTorch (torch.version.hip set),
# pick the .hip / *_hip.* sources committed in this fork instead of the upstream .cu sources.
# On NVIDIA CUDA PyTorch (torch.version.hip is None), keep the upstream .cu sources unchanged.
# The HIP sources were generated from the CUDA ones via `hipify-perl -inplace` and are kept
# committed so a fresh AMD machine can rebuild without re-running hipify.

KERNEL_DIR = "lib/custom_rasterizer_kernel"


def _is_rocm() -> bool:
    """True if PyTorch was built against ROCm (i.e. AMD GPU stack)."""
    return getattr(torch.version, "hip", None) is not None


if _is_rocm() and os.path.exists(os.path.join(KERNEL_DIR, "rasterizer_gpu.hip")):
    sources = [
        os.path.join(KERNEL_DIR, "rasterizer_hip.cpp"),
        os.path.join(KERNEL_DIR, "grid_neighbor_hip.cpp"),
        os.path.join(KERNEL_DIR, "rasterizer_gpu.hip"),
    ]
    print("[custom_rasterizer/setup.py] ROCm PyTorch detected — using HIP sources.")
else:
    sources = [
        os.path.join(KERNEL_DIR, "rasterizer.cpp"),
        os.path.join(KERNEL_DIR, "grid_neighbor.cpp"),
        os.path.join(KERNEL_DIR, "rasterizer_gpu.cu"),
    ]
    if not _is_rocm():
        print("[custom_rasterizer/setup.py] CUDA PyTorch detected — using upstream CUDA sources.")
    else:
        print("[custom_rasterizer/setup.py] ROCm PyTorch but HIP sources missing — falling back to CUDA sources (build will likely fail).")

custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    sources,
)

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={"build_ext": BuildExtension},
)
