"""TorchScript backend for Depth Anything v3 any-view models.

Sibling of ``da3anyview.DA3AnyViewONNX/TRT``: loads a torch.jit-traced
any-view export (fixed ``num_views``, image-only input) and runs it with
plain PyTorch — no ``depth_anything_3`` or xFormers dependency.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .base_da3 import BaseDA3Model

_GEOMETRY_RE = re.compile(r"_(\d+)x(\d+)x(\d+)_")


def _parse_geometry(model_path: str) -> tuple[int, int, int]:
    """Fallback: parse ``<N>x<W>x<H>`` from the file name (num_views, W, H)."""
    m = _GEOMETRY_RE.search(model_path)
    if m is None:
        raise ValueError(
            f"Cannot resolve geometry from '{model_path}': the export stores "
            "num_views/target_h/target_w as module attributes, and the file "
            "name should contain <N>x<W>x<H> (e.g. da3_anyview_24x644x490_...)."
        )
    num_views, width, height = (int(g) for g in m.groups())
    return num_views, width, height


class DA3AnyViewTorchScript(BaseDA3Model):
    """Any-view TorchScript inference: images → depth bundle (numpy in/out)."""

    _OUTPUT_NAMES = ("depth", "depth_conf", "pred_extrinsics", "pred_intrinsics")

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = device
        self.model = torch.jit.load(model_path, map_location=device)
        print(f"[TorchScript] Loaded {model_path}")

        # Geometry is baked into the wrapper module at export time; fall back
        # to the <N>x<W>x<H> file-name convention if the attributes are absent.
        self.num_views = getattr(self.model, "num_views", None)
        self.target_h = getattr(self.model, "target_h", None)
        self.target_w = getattr(self.model, "target_w", None)
        if None in (self.num_views, self.target_h, self.target_w):
            self.num_views, self.target_w, self.target_h = _parse_geometry(model_path)

        self.inputs = [
            {
                "name": "image",
                "shape": [1, self.num_views, 3, self.target_h, self.target_w],
            }
        ]
        self.outputs = [{"name": n, "shape": []} for n in self._OUTPUT_NAMES]
        print(
            f"[TorchScript] num_views={self.num_views}  "
            f"{self.target_w}x{self.target_h}  "
            f"extrinsics input={self.uses_extrinsics}"
        )

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run the traced graph: ``{"image": (1,N,3,H,W) fp32}`` → output dict."""
        image = torch.from_numpy(feed["image"].astype(np.float32)).to(self.device)
        with torch.inference_mode():
            outputs = self.model(image)
        return dict(zip(self._OUTPUT_NAMES, [o.float().cpu().numpy() for o in outputs]))
