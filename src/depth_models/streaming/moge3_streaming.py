#!/usr/bin/env python3
"""
torch.export multi-camera streaming — MoGe v3 (moge3) RGB backend.

The RGB-only counterpart of ``A2F_Streaming``: streams N synchronized RGB
views through ``Moge3_NestedPT2`` (the DA3 any-view graph composed with
MoGe v3, whose monocular metric depth replaces the Any2Full metric branch
— and with it the sensor-depth prompt Any2Full needed, so there are no
``depth_dirs``).  The RGB folders are the ``input_dirs``.  MoGe v3 runs
per view on each chunk's RGB images, so every chunk carries ``num_views``
RGB images and the any-view export sees exactly its fixed view count.

``chunk_size`` is the any-view export's fixed ``num_views``: every chunk
runs the any-view graph once (predicting its own camera poses and
intrinsics, so no camera parameters are passed) plus one MoGe v3
inference per view (the metric anchor the any-view depth is aligned to).
Outputs are saved per time step under ``depth_<RGB folder name>/``, in
the same .lz4 depth + .npz pose format as the other backends.

Run through ``tools/general_test/pipeline/run_depth_stream.py --backend moge3``.
"""

from __future__ import annotations

from .base_streaming import BaseStreaming

from depth_models.moge3.moge3nested import (
    Moge3_NestedPT2,
    _DEFAULT_INTRINSICS_SOURCE,
)

# ---------------------------------------------------------------------------
# Default model paths (any-view fixed num_views = 64, no extrinsics input)
# ---------------------------------------------------------------------------
_DEFAULT_ANYVIEW = "weights/da3/da3_anyview_64x644x490_giant-large-1.1_bf16.pt2"
_DEFAULT_MOGE3 = "weights/moge3/moge3_l.pt2"


class Moge3_Streaming(BaseStreaming):
    """MoGe v3 RGB streaming: input_dirs = RGB folders (no depth inputs).

    MoGe v3 requires CUDA (the exported graph is fp16 and the sparse
    refiner uses Triton kernels), so ``device`` defaults to cuda and a CPU
    request fails inside ``MoGev3_PT2`` with a clear error.
    """

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
        moge3_model_path: str | None = None,
        refiner_path: str | None = None,
        compile: bool = True,
        refine_steps: int = 3,
        mask_dirs: list[str] | None = None,
        intrinsics_source: str = _DEFAULT_INTRINSICS_SOURCE,
    ):
        self.anyview_model_path = anyview_model_path or _DEFAULT_ANYVIEW
        self.moge3_model_path = moge3_model_path or _DEFAULT_MOGE3
        self.refiner_path = refiner_path
        self.compile = compile
        self.refine_steps = refine_steps
        self.intrinsics_source = intrinsics_source
        self.device = device
        # chunk_size is the any-view export's fixed num_views: each chunk
        # carries num_views RGB images, so the model must load before the
        # base __init__.
        self._load_model()
        super().__init__(
            input_dirs=input_dirs,
            save_dir=save_dir,
            config=config,
            chunk_size=self.model_num_views,
            start_frame=start_frame,
            max_frames=max_frames,
            interval=interval,
            device=device,
            mask_dirs=mask_dirs,
        )
        print("Moge3_Streaming init done.")

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        print(f"Loading torch.export any-view model: {self.anyview_model_path}")
        print(f"Loading MoGe v3 PT2: {self.moge3_model_path}")
        self.model = Moge3_NestedPT2(
            self.anyview_model_path,
            self.moge3_model_path,
            refiner_path=self.refiner_path,
            device=self.device or "cuda",
            compile=self.compile,
            refine_steps=self.refine_steps,
            intrinsics_source=self.intrinsics_source,
        )

        # A chunk is num_views RGB images for the fixed-N any-view graph;
        # MoGe v3 runs per view as the metric anchor.
        self.model_num_views = self.model.num_views
        print("Model parameters:")
        print(f"  chunk_size : {self.model_num_views}  (any-view num_views={self.model.num_views})")
        print(f"  image size : {self.model.target_h}x{self.model.target_w}")
        print(f"  camera input: none (image-only; poses + intrinsics predicted by the graph)")
        print("  metric anchor: MoGe v3 per view (monocular metric depth, "
              f"fixed {self.model.moge3.width}x{self.model.moge3.height} graph)")
        print(f"  intrinsics  : {self.model.intrinsics_source} "
              "(anyview = graph-predicted; moge3 = per-view MoGe estimate on the tile grid)")

    def _process_chunk(self, start: int, end: int) -> dict:
        # img_list holds one RGB path per time step — no depth folders (MoGe
        # v3 is monocular and needs no sensor depth prompt).
        rgb_paths = self.img_list[start:end]

        # The nested pipeline preprocesses the views, runs the any-view
        # graph, runs MoGe v3 per view as the metric anchor, aligns the
        # any-view depth to it, crops the padded outputs back to the tile
        # region and un-pads the intrinsics.  The pipeline returns
        # "depth_conf"; the base streaming contract expects "conf".
        result = self.model.infer(rgb_paths)
        result["conf"] = result.pop("depth_conf")
        return result
