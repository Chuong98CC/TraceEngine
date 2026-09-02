"""torch.export (.pt2) backend for Depth Anything v3 any-view models.

Sibling of ``da3anyview.DA3AnyViewONNX/TRT``: loads a fixed-``num_views``
``torch.export`` program (image-only input) and runs it with plain PyTorch —
no ``depth_anything_3`` or xFormers dependency.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .base_da3 import BaseDA3Model

# da3_anyview_<N>x<W>x<H>_<model>.pt2 — N is the fixed view count.
_GEOMETRY_RE = re.compile(r"_(\d+)x(\d+)x(\d+)_")


def _parse_geometry(model_path: str) -> tuple[int, int, int]:
    """Parse ``<N>x<W>x<H>`` from the file name → (num_views, W, H).

    torch.export does not carry plain Python attributes from the wrapper into
    the artifact, so the file name is the single source of geometry.
    """
    m = _GEOMETRY_RE.search(model_path)
    if m is None:
        raise ValueError(
            f"Cannot resolve geometry from '{model_path}': the file name should "
            "contain <N>x<W>x<H> (fixed-N), e.g. "
            "da3_anyview_24x644x490_giant-large-1.1.pt2."
        )
    n_str, width, height = m.groups()
    return int(n_str), int(width), int(height)


class DA3AnyViewPT2(BaseDA3Model):
    """Any-view torch.export inference: (ImageInput in, numpy out)."""

    _OUTPUT_NAMES = ("depth", "depth_conf", "pred_extrinsics", "pred_intrinsics")

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        compile: bool = True,
    ) -> None:
        self.device = device
        program = torch.export.load(model_path)
        module = program.module()
        if compile:
            # Inductor codegen for the exported graph (the graph is fixed-N,
            # so dynamic=True only keeps the codegen shape-generic).
            module = torch.compile(module, dynamic=True)
        self.model = module
        print(f"[PT2] Loaded {model_path} (compile={compile})")

        self.num_views, self.target_w, self.target_h = _parse_geometry(model_path)
        print(f"[PT2] num_views={self.num_views}  {self.target_w}x{self.target_h}")

    def run(self, img_batch: torch.Tensor) -> dict[str, np.ndarray]:
        """Run the exported graph on the preprocessed ``(1,N,3,H,W)`` fp32
        CPU tensor batch."""
        image = img_batch.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return dict(zip(self._OUTPUT_NAMES, [o.float().cpu().numpy() for o in outputs]))

    def infer(self, imgs: list) -> dict:
        """Run the any-view model on *N* image paths, RGB arrays or tensors.

        Returns the mapped output dict with batched values (``(1,N,…)``): the
        graph predicts its own camera poses and intrinsics, so no camera
        parameters are consumed.
        """
        img_batch, _ = self.preprocess_views(imgs)
        raw = self.run(img_batch)
        return self.map_anyview_keys(raw)
