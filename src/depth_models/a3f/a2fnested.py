"""Nested Any2Full + DA3 any-view pipeline (RGB + raw depth -> aligned depth).

``A2F_NestedPT2`` is the a3f counterpart of ``DA3NestedPT2``: the any-view
component stays the DA3 any-view graph (fixed ``num_views``, image-only, poses
and intrinsics predicted in-graph), while the **metric component is swapped
for Any2Full** (``Any2Full_PT2``).  Any2Full receives the RGB image plus the
sensor's sparse metric depth as a prompt and returns an *enhanced dense depth*,
which replaces the metric-depth branch that DA3NestedPT2 feeds to the
alignment (least-squares scale fit + sky handling).

Two structural differences from ``DA3NestedPT2``:

- The metric branch no longer runs on the any-view letterbox grid — Any2Full
  is a fixed-resolution graph (480x640) that resizes its inputs internally, so
  each view is processed from its **original** RGB + raw depth, and the
  enhanced depth is resized back to the view's tile and letterbox-padded onto
  the any-view grid for alignment (the meta from the any-view preprocess
  provides the tile scale and pad).
- The enhanced depth is **already metric metres** (Any2Full's affine fit
  against the sparse anchors), so the alignment skips its focal/300 metric
  rescale (``apply_metric_scaling_step=False``) — DA3's raw network depth
  needs that step, Any2Full's does not.

Any2Full has no sky output, so the alignment's sky mask is all non-sky and the
sky-region clamp becomes a no-op.

With ``use_depth_enhance=False`` the Any2Full model is not loaded at all: the
**raw sensor depth** is placed on the any-view grid directly (resized to the
view's tile and letterbox-padded, invalid pixels kept at 0) and fed to the
alignment.  The metric component then *is* the raw sparse depth, so the
alignment's LSQ fit runs over the valid sensor pixels only (the alignment's
``metric_depth > 1e-2`` mask drops the zeros).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from depth_models.da3.model.da3anyview import DA3AnyViewPT2
from depth_models.da3.model.da3nested import DA3NestedPT2
from utils.file_io.image_io import ImageInput

from .any2full import Any2Full_PT2


class A2F_NestedPT2(DA3NestedPT2):
    """Any2Full (enhance) + DA3 any-view pipeline: RGB + raw depth -> aligned depth.

    The any-view sub-model is loaded from a separate .pt2 artifact exactly like
    ``DA3NestedPT2``; the metric component is ``Any2Full_PT2``, so the
    enhance/scale component can be swapped without re-exporting the any-view
    graph.  ``num_views``/``target_h``/``target_w`` come from the any-view
    export; Any2Full keeps its own fixed 480x640 input.
    """

    def __init__(
        self,
        anyview_path: str,
        a2f_path: str,
        device: str = "cuda",
        compile: bool = True,
        init_scaling: bool = True,
        min_depth: float = 0.0,
        max_depth: float = 1e3,
        use_depth_enhance: bool = True,
    ) -> None:
        self.av = DA3AnyViewPT2(anyview_path, device, compile=compile)
        self.use_depth_enhance = use_depth_enhance
        self.a2f = None
        if use_depth_enhance:
            self.a2f = Any2Full_PT2(
                a2f_path,
                device=device,
                init_scaling=init_scaling,
                max_depth=max_depth,
                min_depth=min_depth,
            )
        # BaseDA3Model methods (align_with_metric, crop) need the any-view size.
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        self.num_views = self.av.num_views  # fixed view count of the any-view graph
        if self.use_depth_enhance:
            print(
                f"[A2F-NESTED-PT2] anyview N={self.num_views} @ {self.target_h}x"
                f"{self.target_w}, a2f @ {self.a2f.target_h}x{self.a2f.target_w} "
                f"(fixed graph, inputs resized internally)"
            )
        else:
            print(
                f"[A2F-NESTED-PT2] anyview N={self.num_views} @ {self.target_h}x"
                f"{self.target_w}, depth enhance DISABLED "
                "(raw depth fed directly to the alignment)"
            )

    def infer(self, imgs: list[ImageInput],
              depths: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Run the full nested pipeline; returns cropped numpy outputs.

        ``imgs`` are the ``num_views`` RGB views (paths, RGB arrays or
        tensors); ``depths`` the parallel raw metric depth maps ``(H, W)``
        float32 metres with 0 = invalid (the Any2Full prompt).  The any-view
        graph predicts its own camera poses AND intrinsics (image-only
        input), so no camera parameters are consumed; the output keeps the
        predicted poses/intrinsics, like ``DA3NestedPT2.infer``.
        """
        n = len(imgs)
        if len(depths) != n:
            raise ValueError(
                f"Got {len(imgs)} RGB views but {len(depths)} depth maps; "
                "pass one raw depth map per view."
            )
        if self.av.num_views is not None and n != self.av.num_views:
            raise ValueError(
                f"Got {n} views but the any-view PT2 model expects "
                f"{self.av.num_views}. Pass exactly {self.av.num_views} views."
            )

        # 1. Any-view: letterbox preprocess + run + map.
        img_batch, metas = self.av.preprocess_views(imgs)
        av = self.av.map_anyview_keys(self.av.run(img_batch))

        # 2. Metric branch on the any-view grid: per-view enhanced depth
        #    (Any2Full) or the raw sensor depth when enhance is disabled.
        if self.use_depth_enhance:
            metric_depths, metric_skys = self._run_metric_branch(imgs, depths, metas)
        else:
            metric_depths, metric_skys = self._grid_raw_depths(depths, metas)

        # 3. Align any-view depth to the metric depth (already metric — skip
        #    the focal rescale; sky is absent, so all non-sky).
        result = self.align_with_metric(av, metric_depths, metric_skys,
                                        apply_metric_scaling_step=False)

        # 4. Crop padded outputs back to the tile region; un-pad intrinsics
        return self._crop_result(result, metas)

    def _run_metric_branch(
        self,
        imgs: list,
        depths: list[np.ndarray],
        metas: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-view Any2Full densification -> enhanced depth on the any-view grid.

        Each view is fed to Any2Full at its **original** resolution (the graph
        resizes RGB/depth to its fixed 480x640 internally), the enhanced depth
        is resized back to the view's tile (``meta["scale_factor"]``) and
        letterbox-padded onto the any-view grid (``meta["pad_top/left"]``), so
        the alignment sees matching pixels.  Pad regions are zeroed (excluded
        by the alignment's ``metric_depth > 1e-2`` mask).  Returns
        ``(1, N, H_av, W_av)`` enhanced depth + all-zero sky.
        """
        h, w = self.av.target_h, self.av.target_w
        n = len(imgs)
        grid = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            # np.ascontiguousarray: a numpy view with negative strides (e.g. a
            # channel-flipped slice) is rejected by torch.from_numpy, which
            # Any2Full's preprocess applies to the depth map (the RGB view is
            # decoded by image_io — any ImageInput — inside that preprocess).
            # Callers pass contiguous RGB arrays, so the guard is a harmless
            # no-op kept as insurance for raw ndarray inputs.
            rgb_t, dep_t = self.a2f.preprocess(
                np.ascontiguousarray(imgs[i]) if isinstance(imgs[i], np.ndarray) else imgs[i],
                depths[i],
            )
            with torch.inference_mode():
                depth, disparity_pre, prompt = self.a2f.infer(rgb_t, dep_t)
                pred = self.a2f.postprocess(depth, disparity_pre, prompt)
            pred = pred.squeeze(0).squeeze(0).cpu().numpy()
            # tile = original view scaled by the any-view letterbox scale; the
            # enhanced depth corresponds to the original pixels, so resize back
            grid[0, i] = self._place_on_grid(pred, metas[i], h, w)
        skys = np.zeros_like(grid)
        return grid, skys

    @staticmethod
    def _place_on_grid(depth: np.ndarray, meta: dict, h: int, w: int) -> np.ndarray:
        """Resize a per-view depth (original resolution) to its tile and pad it
        onto the any-view grid, matching the letterbox geometry of the view's
        RGB.  Pad regions are zeroed (excluded by the alignment's
        ``metric_depth > 1e-2`` mask)."""
        tile = cv2.resize(depth, (meta["tile_w"], meta["tile_h"]),
                          interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(
            tile,
            meta["pad_top"],
            h - meta["pad_top"] - meta["tile_h"],
            meta["pad_left"],
            w - meta["pad_left"] - meta["tile_w"],
            cv2.BORDER_CONSTANT,
            value=0,
        )

    def _grid_raw_depths(
        self,
        depths: list[np.ndarray],
        metas: list[dict],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Raw sensor depth on the any-view grid (depth enhance disabled).

        Each raw depth map (original resolution, metres, 0 = invalid) is
        resized to its view's tile and letterbox-padded onto the any-view
        grid, exactly like the enhanced depth.  Invalid pixels stay 0 and are
        excluded by the alignment's ``metric_depth > 1e-2`` mask, so the
        alignment's LSQ fit runs over the valid sensor pixels only.  Returns
        ``(1, N, H_av, W_av)`` raw depth + all-zero sky.
        """
        h, w = self.av.target_h, self.av.target_w
        n = len(depths)
        grid = np.zeros((1, n, h, w), dtype=np.float32)
        for i in range(n):
            grid[0, i] = self._place_on_grid(depths[i], metas[i], h, w)
        skys = np.zeros_like(grid)
        return grid, skys
