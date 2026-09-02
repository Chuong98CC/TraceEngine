"""Sparse 3D UNet refiner of MoGe v3, vendored so that `infer_pt2` is a
standalone package with no dependency on the `moge` source tree.

The refiner cannot be traced by torch.export (custom flex_gemm Triton
kernels, data-dependent voxel shapes, Python-object NeighborCache state),
so it runs eagerly at inference time from its companion checkpoint
(`moge3_l_refiner.pt`). `flex_gemm` itself remains an external dependency.
"""

from .sparse_unet import Sparse3DUNet

__all__ = ['Sparse3DUNet']
