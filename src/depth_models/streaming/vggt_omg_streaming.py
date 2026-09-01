#!/usr/bin/env python3
"""
VGGT-Omega multi-camera streaming — exported-program backend.

Implements the two backend hooks of ``base_streaming.BaseStreaming``:
loading the pre-exported VGGT-Omega model and running chunks through it.
The torch.export program (``.pt2`` / ``.pt``) has a FIXED view count;
``chunk_size`` is derived from the export's ``num_views``, so the model
path is the only model-related parameter.  A short final chunk is padded
with images from the previous chunk (or duplicated when the whole sequence
is shorter than one chunk); the padded outputs are discarded.

``VGGT_Omega`` performs its own letterbox preprocessing, crops the
spatial outputs back to the content region, and already shifts the principal
point by each frame's letterbox offset — so mapping the outputs to the base
data contract is just key renaming (``depth_conf`` → ``conf``).

Run through ``tools/run_depth_stream.py --backend vggt_omega``.
"""

from __future__ import annotations

import numpy as np

from depth_models.vggt_omega.vggt_omega import VGGT_Omega

from .base_streaming import BaseStreaming

# ---------------------------------------------------------------------------
# Default model path (fixed num_views = 64)
# ---------------------------------------------------------------------------
_MODEL_PATH = "weights/vggt_omg/vggt_omg_64x640x480_bf16.pt2"


# ===========================================================================
# Backend
# ===========================================================================
class VGGT_OMG_Streaming(BaseStreaming):
    """VGGT-Omega multi-camera streaming with N synchronized image folders."""

    def __init__(
        self,
        input_dirs: list[str],
        save_dir: str,
        config: dict,
        start_frame: int = 0,
        max_frames: int | None = None,
        interval: int = 1,
        device: str | None = None,
        model_path: str | None = None,
        mask_dirs: list[str] | None = None,
    ):
        # The exported graph is CUDA-baked (lifted weights moved to cuda at
        # load, runtime tensors sent to cuda), so only cuda makes sense here.
        if device not in (None, "cuda"):
            raise ValueError(
                f"VGGT-Omega exported programs are CUDA-only (device baked at "
                f"trace time); got device={device!r}"
            )
        self.model_path = model_path or _MODEL_PATH
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
        print("VGGT_OMG_Streaming init done.")

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        print(f"Loading VGGT-Omega exported program: {self.model_path}")
        self.model = VGGT_Omega(self.model_path)

        self.model_num_views = self.model.num_views
        print("Model parameters:")
        print(f"  chunk_size : {self.model_num_views}  (fixed num_views from the export)")
        print(f"  image size : {self.model.target_h}x{self.model.target_w}")

    def _process_chunk(self, start: int, end: int) -> dict:
        paths = self.img_list[start:end]
        out = self.model.infer(paths)

        # Mixed aspect ratios in one chunk return per-frame lists; the base
        # run loop needs uniform stacked arrays.
        depth = out["depth"]
        conf = out["depth_conf"]
        if isinstance(depth, list):
            raise ValueError(
                "VGGT-Omega returned per-frame outputs (mixed aspect ratios); "
                "streaming requires uniform image resolutions across the chunk"
            )

        return {
            "depth": depth.astype(np.float32),  # (N, h', w')
            "conf": conf.astype(np.float32),  # (N, h', w')
            "extrinsics": out["extrinsic"].astype(np.float32),  # (N, 3, 4) w2c
            "intrinsics": out["intrinsic"].astype(np.float32),  # (N, 3, 3)
        }
