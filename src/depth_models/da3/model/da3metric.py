"""Metric Depth Anything v3 torch.export wrapper (single-image, raw depth + sky).

Standalone copy of ``tools/model/da3metric.py`` containing only the
torch.export (``DA3MetricPT2``) path.  Metric depth in metres is a caller-side
``focal * depth / 300`` step (the model returns raw network depth + sky).
"""

from __future__ import annotations

import re

import numpy as np
import torch

from .base_da3 import BaseDA3Model

# da3_metric_<W>x<H>_<model>.pt2 — resolution baked into the file name.
_GEOMETRY_RE = re.compile(r"_(\d+)x(\d+)_")


_PROBE_RESOLUTION = (490, 644)  # (H, W) tried first; the metric export default


def _probe_input_hw(module: torch.nn.Module, device: str) -> tuple[int, int]:
    """Infer the graph's fixed input resolution with a dummy forward.

    Used when the file name carries no ``<W>x<H>`` (e.g. a swapped metric
    model).  The exported graph is fixed-resolution (its guards demand the
    exact input shape), so the probe simply verifies that the default
    resolution is accepted and fails with a clear error otherwise.
    """
    h, w = _PROBE_RESOLUTION
    dummy = torch.zeros(1, 3, h, w, device=device)
    try:
        with torch.no_grad():
            module(dummy)
        return h, w
    except Exception as e:
        raise ValueError(
            f"Cannot resolve the input resolution of '{getattr(module, '_name', 'graph')}': "
            f"the fixed-N torch.export graph rejects {w}x{h}. Exported metric graphs are "
            f"fixed-resolution; name the file da3_metric_<W>x<H>_... or re-export at a "
            f"known resolution."
        ) from e


class DA3MetricPT2(BaseDA3Model):
    """Metric torch.export inference on a single already-preprocessed view."""

    _OUTPUT_NAMES = ("depth", "sky")

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
            # Inductor codegen for the exported graph.
            module = torch.compile(module, dynamic=True)
        self.model = module
        print(f"[PT2] Loaded {model_path} (compile={compile})")

        m = _GEOMETRY_RE.search(model_path)
        if m is not None:
            width, height = m.groups()
            self.target_w, self.target_h = int(width), int(height)
        else:
            # File name carries no <W>x<H> (e.g. a swapped metric model) —
            # resolve the input resolution by running a small dummy forward.
            self.target_h, self.target_w = _probe_input_hw(module, device)
        self.num_views = None  # single-view model

        self.inputs = [{"name": "image", "shape": [1, 3, self.target_h, self.target_w]}]
        self.outputs = [{"name": n, "shape": []} for n in self._OUTPUT_NAMES]
        print(f"[PT2] metric @ {self.target_w}x{self.target_h}")

    def run(self, feed: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Run the exported graph: ``{"image": (1,3,H,W) fp32 tensor}``."""
        image = feed["image"].to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return dict(zip(self._OUTPUT_NAMES, [o.float().cpu().numpy() for o in outputs]))

    def infer_view(
        self, img: torch.Tensor, apply_mono_sky: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW fp32 tensor →
        ``(depth, sky)`` numpy."""
        raw = self.run({"image": img.unsqueeze(0)})
        depth, sky = self.extract_metric(raw)
        if apply_mono_sky:
            depth = self.apply_mono_sky(depth, sky)
        return depth, sky
