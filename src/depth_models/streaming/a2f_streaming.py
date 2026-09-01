#!/usr/bin/env python3
"""
torch.export multi-camera streaming — Any2Full (a3f) RGB-D backend.

The a3f counterpart of ``DA3_Streaming``: streams N synchronized RGB views
plus their raw sensor depth through ``A2F_NestedPT2`` (the DA3 any-view graph
composed with Any2Full, which densifies the sparse depth).  The RGB folders
are the ``input_dirs``; the parallel raw-depth folders (one per RGB folder,
``.lz4`` uint16 mm maps with matching frame stems, e.g. ``cam_head`` →
``depth_cam_head``) are passed as ``depth_dirs`` and loaded via
``load_depth_lz4`` (metres = mm x ``depth_scale``, 0 = invalid).  The
online streamer (``tools/astribot/run_subtask_stream.py --backend a2f``)
supplies the depth as raw uint16 ndarrays instead — ``_process_chunk`` and
``_load_depth`` accept both.

``chunk_size`` is the any-view export's fixed ``num_views``: every chunk
carries ``num_views`` RGB images and ``num_views`` raw-depth maps (the
any-view graph sees exactly its fixed view count, the depth maps are the
Any2Full prompt).  The any-view export predicts its own camera poses and
intrinsics, so no camera parameters are passed.  Outputs are saved per time
step under ``depth_<RGB folder name>/``, in the same .lz4 depth + .npz pose
format as the other backends.

With ``use_depth_enhance=False`` (``run_depth_stream.py --no-depth-enhance``) the
Any2Full model is not loaded and the raw sensor depth is fed directly to the
alignment as the metric component — the input contract (RGB folders +
``depth_dirs``) is unchanged.

Run through ``tools/general_test/pipeline/run_depth_stream.py --backend a2f``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .base_streaming import BaseStreaming

from depth_models.a3f.a2fnested import A2F_NestedPT2
from utils.astribot_dataloader import load_depth_lz4

# ---------------------------------------------------------------------------
# Default model paths (any-view fixed num_views = 64, no extrinsics input)
# ---------------------------------------------------------------------------
_DEFAULT_ANYVIEW = "weights/da3/da3_anyview_64x644x490_giant-large-1.1_bf16.pt2"
_DEFAULT_A2F = "weights/any2full/Any2Full_vitl_bf16.pt2"


class A2F_Streaming(BaseStreaming):
    """Any2Full RGB-D streaming: input_dirs = RGB folders, depth_dirs parallel."""

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
        a2f_model_path: str | None = None,
        compile: bool = True,
        mask_dirs: list[str] | None = None,
        depth_dirs: list[str] | None = None,
        depth_scale: float = 0.001,
        init_scaling: bool = True,
        min_depth: float = 0.0,
        max_depth: float = 1e3,
        use_depth_enhance: bool = True,
    ):
        self.anyview_model_path = anyview_model_path or _DEFAULT_ANYVIEW
        self.a2f_model_path = a2f_model_path or _DEFAULT_A2F
        self.compile = compile
        self.init_scaling = init_scaling
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_depth_enhance = use_depth_enhance
        self.depth_scale = depth_scale
        # (H, W) of the raw depth maps, derived from the paired RGB on the
        # first chunk (all frames share the same resolution).
        self._depth_shape: tuple[int, int] | None = None
        self.device = device
        # chunk_size is the any-view export's fixed num_views: each chunk
        # carries num_views RGB images (and num_views parallel depth maps,
        # which do not count toward num_cams), so the model must load before
        # the base __init__.
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
            depth_dirs=depth_dirs,
        )
        if self.depth_dirs is None:
            raise ValueError(
                "The a2f backend requires one --depth-dirs folder per "
                "--input-dirs folder (raw sensor depth, .lz4 uint16 mm)."
            )
        print("A2F_Streaming init done.")

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        print(f"Loading torch.export any-view model: {self.anyview_model_path}")
        if self.use_depth_enhance:
            print(f"Loading Any2Full PT2: {self.a2f_model_path}")
        else:
            print("Depth enhance disabled: raw depth fed directly to the alignment")
        self.model = A2F_NestedPT2(
            self.anyview_model_path,
            self.a2f_model_path,
            device=self.device or "cuda",
            compile=self.compile,
            init_scaling=self.init_scaling,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            use_depth_enhance=self.use_depth_enhance,
        )

        # A chunk is num_views RGB images for the fixed-N any-view graph;
        # the parallel raw-depth maps are the Any2Full prompt.
        self.model_num_views = self.model.num_views
        print("Model parameters:")
        print(f"  chunk_size : {self.model_num_views}  (any-view num_views={self.model.num_views})")
        print(f"  image size : {self.model.target_h}x{self.model.target_w}")
        print(f"  camera input: none (image-only; poses + intrinsics predicted by the graph)")
        if self.use_depth_enhance:
            print(f"  depth_dirs: raw depth .lz4 (uint16 mm x {self.depth_scale} -> metres); Any2Full densifies it")
        else:
            print(f"  depth_dirs: raw depth .lz4 (uint16 mm x {self.depth_scale} -> metres); fed directly to the alignment")

    def _process_chunk(self, start: int, end: int) -> dict:
        # img_list holds one RGB path per time step; depth_paths is the
        # parallel raw-depth list built by _load_depth_paths.
        rgb_paths = self.img_list[start:end]
        if self._depth_shape is None:
            # Online mode swaps BGR arrays into img_list (see
            # tools/astribot/run_subtask_stream.py), so the reference frame
            # may be an array instead of a path.
            first = rgb_paths[0]
            if isinstance(first, str):
                img = cv2.imread(first, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise RuntimeError(f"Could not read {first} to derive the depth shape")
            else:
                img = first
            self._depth_shape = tuple(img.shape[:2])
        depths = [
            self._load_depth(p, self._depth_shape, self.depth_scale)
            for p in self.depth_paths[start:end]
        ]

        # The nested pipeline preprocesses the views, runs the any-view graph,
        # densifies each view with Any2Full (RGB + raw depth), aligns the
        # any-view depth to the enhanced depth, crops the padded outputs back
        # to the tile region and un-pads the intrinsics.  The pipeline returns
        # "depth_conf"; the base streaming contract expects "conf".
        result = self.model.infer(rgb_paths, depths)
        result["conf"] = result.pop("depth_conf")
        return result

    def _out_view_slots(self) -> list[int]:
        """Only the RGB folders emit depth — the depth folders are inputs."""
        return [0]

    @staticmethod
    def _load_depth(path_or_arr, shape: tuple[int, int], scale: float) -> np.ndarray:
        """Raw sensor depth -> (H, W) float32 metres.

        Accepts an .lz4 uint16 mm path (disk flow) or a raw uint16 ndarray
        (online mode; see tools/astribot/run_subtask_stream.py); 0 = invalid
        -> 0.0 metres."""
        if isinstance(path_or_arr, (str, Path)):
            depth = load_depth_lz4(Path(path_or_arr), shape).astype(np.float32)
        else:
            depth = np.asarray(path_or_arr).astype(np.float32)
        depth *= scale
        depth[depth <= 0.0] = 0.0
        return depth
