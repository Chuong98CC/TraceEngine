"""Minimal standalone inference for the exported MoGe v3 pt2 checkpoint.

Preprocessing, inference and post-processing only -- no export code, no
visualization, no mesh/ply saving. The model code is vendored so the
package is standalone: the sparse refiner lives in `refiner/` (copied from
`moge/model/modules/`) and the geometry post-processing in
`utils/geometry_torch.py` (copied from `moge/utils/`). External pip
dependencies are `torch`, `flex_gemm`, `utils3d[_moge]`, `opencv-python`,
`numpy`, `scipy`.

Usage:
    from minimal_infer_pt2.moge_pt2 import MoGev3_PT2
    model = MoGev3_PT2('weights/moge3/moge3_l.pt2')   # + moge3_l_refiner.pt
    out = model.infer(image_bgr)                      # {points, depth, mask, intrinsics, normal}

Also see `minimal_infer_pt2/demo.py` for a runnable example.
"""

__version__ = '1.0.0'
