#!/usr/bin/env python3
"""
torch.export multi-camera streaming — generalized N-folder backend.

Implements the two backend hooks of ``base_streaming.BaseStreaming``:
loading the pre-exported DA3 torch.export programs and running chunks
through them.  DA3 is split into two independent checkpoints — the any-view
graph (FIXED view count) and the metric-depth graph — composed by
``DA3NestedPT2`` so the metric component can be swapped without re-exporting
the any-view graph.  Both graphs are image-only (the any-view export
predicts its own camera poses and intrinsics), so chunks are fed as image
paths and no camera parameters are passed.  ``chunk_size`` is derived from
the any-view export's ``num_views``; the two model paths are the only
model-related parameters.  A
short final chunk is padded with images from the previous chunk (or
duplicated when the whole sequence is shorter than one chunk); the padded
outputs are discarded.

Run through ``tools/general_test/run_stream.py --backend da3``.
"""

from __future__ import annotations

import numpy as np

from .base_streaming import BaseStreaming

from depth_models.da3.model.da3nested import DA3NestedPT2

# ---------------------------------------------------------------------------
# Default model paths (any-view fixed num_views = 64, no extrinsics input)
# ---------------------------------------------------------------------------
_DEFAULT_ANYVIEW = "weights/da3/da3_anyview_64x644x490_giant-large-1.1_bf16.pt2"
_DEFAULT_METRIC = "weights/da3/da3_metric_644x490_giant-large-1.1_bf16.pt2"

# ===========================================================================
# Backend
# ===========================================================================
class DA3_Streaming(BaseStreaming):
    """torch.export multi-camera streaming with N synchronized image folders."""

    def __init__(
        self,
        input_dirs: list[str],
        save_dir: str,
        config: dict,
        start_frame: int = 0,
        max_frames: int | None = None,
        interval: int = 1,
        device: str | None = None,
        anyview_model_path: str | None = None,
        metric_model_path: str | None = None,
        compile: bool = True,
        mask_dirs: list[str] | None = None,
    ):
        self.anyview_model_path = anyview_model_path or _DEFAULT_ANYVIEW
        self.metric_model_path = metric_model_path or _DEFAULT_METRIC
        self.compile = compile
        self.device = device
        # chunk_size is derived from the export's fixed num_views, so the
        # model must load before the base __init__ (which needs chunk_size).
        self._load_model()
        super().__init__(
            input_dirs=input_dirs,
            save_dir=save_dir,
            config=config,
            chunk_size=self.model.num_views,
            start_frame=start_frame,
            max_frames=max_frames,
            interval=interval,
            device=device,
            mask_dirs=mask_dirs,
        )
        print("DA3_Streaming init done.")

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        print(f"Loading torch.export any-view model: {self.anyview_model_path}")
        print(f"Loading torch.export metric-depth model: {self.metric_model_path}")
        self.model = DA3NestedPT2(
            self.anyview_model_path,
            self.metric_model_path,
            device=self.device or "cuda",
            compile=self.compile,
        )

        # chunk_size is derived from the export's fixed num_views — there is
        # no user-facing chunk_size parameter for this backend.
        self.model_num_views = self.model.num_views
        print("Model parameters:")
        print(f"  chunk_size : {self.model_num_views}  (fixed num_views from the export)")
        print(f"  image size : {self.model.target_h}x{self.model.target_w}")
        print(f"  camera input: none (image-only; poses + intrinsics predicted by the graph)")

    def _process_chunk(self, start: int, end: int) -> dict:
        paths = self.img_list[start:end]

        # The nested pipeline preprocesses the views, runs the any-view graph,
        # runs the metric graph per view, aligns any-view depth to the metric
        # depth, crops the padded outputs back to the tile region, and un-pads
        # the intrinsics — no camera parameters are fed (the any-view graph is
        # the image-only export and predicts its own poses and intrinsics).
        # The pipeline returns "depth_conf"; the base streaming contract and
        # its consumers (alignment pair, mask stacking) expect "conf".
        result = self.model.infer(paths)
        result["conf"] = result.pop("depth_conf")
        return result
