"""PT2-format (``torch.export``) tooling for WAFTv2 optical flow inference.

Contents:
    - ``export_pt2.py``        Export a WAFTv2 checkpoint to a bf16 ``.pt2``
                               artifact (mirrors ``export_onnx.py``).
    - ``waftv2_pt2.py``        :class:`WAFTv2_PT2` — loads the artifact and
                               does pre/post-processing (mirrors
                               ``model/waft_onnx.py`` + ``model/waft_base.py``).
    - ``infer_image_pt2.py``   CLI: run optical flow on one image pair.
    - ``utils/``               Helper functions backing the above.

The exported artifact is a static-shape ExportedProgram: it expects two RGB
images in [0, 255] at the exported resolution (``bfloat16``) and returns a
single flow tensor ``[1, 2, H, W]``.
"""

import os
import sys

# Make the WAFT repo root importable whenever any infer_pt2 submodule is
# imported.  Entry scripts run as ``python infer_pt2/<script>.py`` have
# sys.path[0] = <repo>/infer_pt2, so ``import config.parser`` / ``import utils...``
# would otherwise fail.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
