"""WAFTv2 optical-flow inference on a ``torch.export`` (``.pt2``) artifact.

Contents:
    - ``waftv2_pt2.py``   :class:`WAFTv2_PT2` — loads the bf16 artifact and
                           does pre/post-processing via the shared
                           ``utils.image_io`` helpers (RGB ``ImageInput``,
                           tensor-first letterbox).

The exported artifact is a static-shape ExportedProgram: it expects two RGB
images in [0, 255] at the exported resolution (``bfloat16``) and returns a
single flow tensor ``[1, 2, H, W]``.
"""
