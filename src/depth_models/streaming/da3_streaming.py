#!/usr/bin/env python3
"""
TorchScript multi-camera streaming — generalized N-folder backend.

Implements the two backend hooks of ``da3_streaming/base_streaming.py``:
loading the pre-exported any-view TorchScript model and running chunks
through it.  The TorchScript export (``tools/export_torchscript.py``) has a
FIXED view count; ``chunk_size`` is derived from the export's
``num_views``, so the model path is the only model-related parameter.  A
short final chunk is padded with images from the previous chunk (or
duplicated when the whole sequence is shorter than one chunk); the padded
outputs are discarded.

Run through ``da3_streaming/run_stream.py --backend torchscript``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .base_streaming import BaseStreaming

# -- make tools/model/ importable -------------------------------------------------
# _TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
# if _TOOLS_DIR not in sys.path:
#     sys.path.insert(0, _TOOLS_DIR)

from depth_models.da3.base_da3 import BaseDA3Model  # noqa: E402
from depth_models.da3.da3_model import DA3AnyViewTorchScript  # noqa: E402

# ---------------------------------------------------------------------------
# Default model path (fixed num_views = 24, no extrinsics input)
# ---------------------------------------------------------------------------
_MODEL_PATH = "weights/da3/da3_anyview_24x644x490_giant-large-1.1.pt"


# ===========================================================================
# Backend
# ===========================================================================
class DA3_Streaming(BaseStreaming):
    """TorchScript multi-camera streaming with N synchronized image folders."""

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
        self.model_path = model_path or _MODEL_PATH
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
        print(f"Loading TorchScript model: {self.model_path}")
        self.model = DA3AnyViewTorchScript(self.model_path, device=self.device or "cuda")

        # chunk_size is derived from the export's fixed num_views — there is
        # no user-facing chunk_size parameter for this backend.
        self.model_num_views = self.model.num_views
        print("Model parameters:")
        print(f"  chunk_size : {self.model_num_views}  (fixed num_views from the export)")
        print(f"  image size : {self.model.target_h}x{self.model.target_w}")
        print(f"  extrinsics input: {self.model.uses_extrinsics}")

    def _process_chunk(self, start: int, end: int) -> dict:
        paths = self.img_list[start:end]
        n = len(paths)
        dummy_intrs = np.tile(self.dummy_intrinsics[None], (n, 1, 1))

        img_batch, intrs_adj, metas = self.model.preprocess_views(paths, dummy_intrs)
        feed = self.model.build_anyview_feed(img_batch, None, intrs_adj)
        raw = self.model.run(feed)
        result = BaseDA3Model.map_anyview_keys(raw)

        depth = np.stack(
            [BaseDA3Model.crop_to_tile(result["depth"][0, i], metas[i]) for i in range(n)]
        ).astype(np.float32)
        conf = np.stack(
            [BaseDA3Model.crop_to_tile(result["depth_conf"][0, i], metas[i]) for i in range(n)]
        ).astype(np.float32)

        intr = result["intrinsics"][0].copy()
        for i in range(n):
            intr[i, 0, 2] -= metas[i]["pad_left"]
            intr[i, 1, 2] -= metas[i]["pad_top"]

        ext = result["extrinsics"][0]
        if ext.shape[1] == 4:
            ext = ext[:, :3, :]

        return {
            "depth": depth,
            "conf": conf,
            "extrinsics": ext.copy().astype(np.float32),
            "intrinsics": intr.astype(np.float32),
        }
