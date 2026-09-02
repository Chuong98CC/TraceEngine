"""Nested Depth Anything v3 pipeline (any-view + metric + alignment).

``DA3NestedPT2`` composes the torch.export any-view and metric wrappers and
aligns their outputs to reproduce ``NestedDepthAnything3Net``.  Output depth
is left **unmasked** (PyTorch does not confidence-mask depth; only sky regions
are set to max depth inside ``align_anyview_with_metric``).  Standalone copy of
``tools/model/da3nested.py`` containing only the torch.export path.
"""

from __future__ import annotations

import numpy as np
import torch

from .base_da3 import BaseDA3Model
from .da3anyview import DA3AnyViewPT2
from .da3metric import DA3MetricPT2


class DA3NestedPT2(BaseDA3Model):
    """Any-view + metric torch.export pipeline replicating the nested PyTorch model.

    The any-view and metric sub-models are loaded from separate .pt2 artifacts,
    so the metric model can be swapped for any other metric .pt2 (same input
    resolution) without re-exporting the any-view graph.
    """

    def __init__(
        self,
        anyview_path: str,
        metric_path: str,
        device: str = "cuda",
        compile: bool = True,
    ) -> None:
        self.av = DA3AnyViewPT2(anyview_path, device, compile=compile)
        self.metric = DA3MetricPT2(metric_path, device, compile=compile)
        # BaseDA3Model methods (normalize_extrinsics, align_*) need target size.
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        self.num_views = self.av.num_views  # fixed view count of the any-view graph
        print(
            f"[NESTED-PT2] anyview N={self.num_views} @ {self.target_h}x"
            f"{self.target_w}, metric @ {self.metric.target_h}x"
            f"{self.metric.target_w}"
        )

    def infer(self, imgs: list[str | np.ndarray]) -> dict[str, np.ndarray]:
        """Run the full nested pipeline; returns cropped numpy outputs.

        The torch.export any-view graph predicts its own camera poses AND
        intrinsics (image-only input), so no camera parameters are consumed;
        the output keeps the predicted poses/intrinsics, mirroring
        ``DepthAnything3.inference(extrinsics=None)``.
        """
        n = len(imgs)
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view PT2 model expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: letterbox preprocess + run + map.
        img_batch, metas = self.av.preprocess_views(imgs)
        av = self.av.map_anyview_keys(self.av.run(img_batch))

        # 2. Metric branch (reuses the any-view letterboxed grid; sizes must match)
        metric_depths, metric_skys = self._run_metric_branch(img_batch)

        # 3. Align any-view depth to metric (padded grid; sky handling inside)
        result = self.align_with_metric(av, metric_depths, metric_skys)

        # 4. Crop padded outputs back to the tile region; un-pad intrinsics
        return self._crop_result(result, metas)

    def _run_metric_branch(
        self,
        av_img_batch: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-view metric inference on the any-view letterbox grid.

        The metric model must share the any-view target size so the already
        letterboxed any-view padded views can be reused verbatim — no extra
        resize.  Returns ``(1, N, H_av, W_av)``.
        """
        h, w = self.av.target_h, self.av.target_w
        mh, mw = self.metric.target_h, self.metric.target_w
        if (mh, mw) != (h, w):
            raise NotImplementedError(
                "Nested pipeline requires the metric and any-view PT2 models to "
                f"share the input size; got metric {mh}x{mw} vs any-view {h}x{w}. "
                "Re-export the metric model at the any-view resolution."
            )

        n = av_img_batch.shape[1]
        depths = np.zeros((1, n, h, w), dtype=np.float32)
        skys = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            d, s = self.metric.infer_view(av_img_batch[0, i], apply_mono_sky=False)
            depths[0, i] = d
            skys[0, i] = s
        return depths, skys

    def _crop_result(self, result: dict, metas: list) -> dict:
        """Crop padded depth/conf to the tile region and un-pad the intrinsics.

        All views share the source resolution, so their tiles are identical in
        size and stack cleanly.  The predicted intrinsics' principal point is
        shifted back by the pad so it matches the cropped image.
        """
        depth = np.stack(
            [self.av.crop_to_tile(result["depth"][i], metas[i]) for i in range(len(metas))]
        )
        conf = np.stack(
            [self.av.crop_to_tile(result["depth_conf"][i], metas[i]) for i in range(len(metas))]
        )
        intr = result["intrinsics"].copy()
        for i, m in enumerate(metas):
            intr[i, 0, 2] -= m["pad_left"]
            intr[i, 1, 2] -= m["pad_top"]
        return {
            "depth": depth,
            "depth_conf": conf,
            "extrinsics": result["extrinsics"],
            "intrinsics": intr,
        }
