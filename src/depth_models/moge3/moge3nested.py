"""Nested MoGe v3 + DA3 any-view pipeline (RGB -> metric-aligned depth).

``Moge3_NestedPT2`` is the MoGe3 counterpart of ``A2F_NestedPT2``: the
any-view component stays the DA3 any-view graph (fixed ``num_views``,
image-only, poses and intrinsics predicted in-graph), while the metric
component is swapped for MoGe v3 (``MoGev3_PT2``).  MoGe v3 is a
monocular metric-depth model — it consumes only the RGB view (no
sensor-depth prompt, unlike Any2Full) and returns a *dense metric*
depth in metres (recovered via its metric-scale head).  That depth
replaces the metric branch DA3NestedPT2 feeds to the alignment
(least-squares scale fit + sky handling).

Two structural differences from ``DA3NestedPT2``:

- The metric branch no longer runs on the any-view letterbox grid —
  MoGe v3 is a fixed-resolution graph (640x480) that stretches its
  input internally, so each view is processed from its **original**
  RGB and the metric depth is resized back to the view's original
  resolution before it is letterbox-padded onto the any-view grid for
  the alignment (the meta from the any-view preprocess provides the
  tile scale and pad).
- The metric depth is **already metric metres** (MoGe v3's
  metric-scale head), so the alignment skips its focal/300 metric
  rescale (``apply_metric_scaling_step=False``) — DA3's raw network
  depth needs that step, MoGe's does not (same reasoning as Any2Full's
  affine-fit output).

MoGe v3 has no sky output, so the alignment's sky mask is all non-sky
and the sky-region clamp becomes a no-op.  MoGe's object/validity mask
is honoured with the model defaults (``apply_mask``): masked-out pixels
(dynamic / ungrounded regions) come back as +inf and are blanked to 0
before placement, excluding them from the anchor set via the
alignment's ``metric_depth > 1e-2`` mask.

Run through ``tools/general_test/pipeline/run_depth_stream.py --backend moge3``.

The camera intrinsics of the output can come from either model: the
any-view graph's predicted intrinsics (``intrinsics_source="anyview"``,
the default — multi-view consistent) or MoGe v3's per-view estimated
intrinsics (``intrinsics_source="moge3"``, converted from MoGe's
normalized-uv convention to the output tile grid exactly like
``tools/general_test/module/infer_moge3.py`` does for its 640x480 grid).
"""

from __future__ import annotations

import cv2
import numpy as np

from depth_models.da3.model.da3anyview import DA3AnyViewPT2
from depth_models.da3.model.da3nested import DA3NestedPT2

from .moge_pt2 import MoGev3_PT2

#: Camera-intrinsics source for the nested output; flip here to A/B-test
#: "anyview" (multi-view consistent) vs "moge3" (per-view monocular) without
#: touching the callers (both Moge3_NestedPT2 and Moge3_Streaming default to
#: this constant).
_DEFAULT_INTRINSICS_SOURCE = "anyview"


class Moge3_NestedPT2(DA3NestedPT2):
    """MoGe v3 (metric) + DA3 any-view pipeline: RGB views -> aligned depth.

    The any-view sub-model is loaded from a separate .pt2 artifact exactly
    like ``DA3NestedPT2``; the metric component is ``MoGev3_PT2``, so the
    metric model can be swapped without re-exporting the any-view graph.
    ``num_views``/``target_h``/``target_w`` come from the any-view export;
    MoGe v3 keeps its own fixed 640x480 input (any view is stretched to it
    internally).
    """

    def __init__(
        self,
        anyview_path: str,
        moge3_path: str,
        refiner_path: str | None = None,
        device: str = "cuda",
        compile: bool = True,
        refine_steps: int = 3,
        intrinsics_source: str = _DEFAULT_INTRINSICS_SOURCE,
    ) -> None:
        if intrinsics_source not in ("anyview", "moge3"):
            raise ValueError(
                f"intrinsics_source must be 'anyview' or 'moge3', got "
                f"{intrinsics_source!r}"
            )
        self.intrinsics_source = intrinsics_source
        self.av = DA3AnyViewPT2(anyview_path, device, compile=compile)
        self.moge3 = MoGev3_PT2(
            moge3_path,
            refiner_path=refiner_path,
            device=device,
            refine_steps=refine_steps,
        )
        # BaseDA3Model methods (align_with_metric, crop) need the any-view size.
        self.target_h = self.av.target_h
        self.target_w = self.av.target_w
        self.num_views = self.av.num_views  # fixed view count of the any-view graph
        print(
            f"[MOGE3-NESTED-PT2] anyview N={self.num_views} @ {self.target_h}x"
            f"{self.target_w}, moge3 @ {self.moge3.width}x{self.moge3.height} "
            f"(fixed graph, inputs stretched internally), "
            f"intrinsics={self.intrinsics_source}"
        )

    def infer(self, imgs: list) -> dict:
        """Run the full nested pipeline; returns cropped numpy outputs.

        ``imgs`` are the ``num_views`` RGB views (paths, RGB arrays or CHW
        tensors).  The any-view graph predicts its own camera poses AND
        intrinsics (image-only input), so no camera parameters are consumed;
        the output keeps the predicted poses (any-view) and the intrinsics of
        ``self.intrinsics_source`` (any-view graph or per-view MoGe v3).
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

        # 2. Metric branch on the any-view grid: per-view MoGe v3 depth
        #    (+ per-view MoGe intrinsics on the tile grid when selected).
        metric_depths, metric_skys, tile_intrinsics = self._run_metric_branch(
            imgs, metas
        )

        # 3. Align any-view depth to the metric depth (already metric — skip
        #    the focal rescale; sky is absent, so all non-sky).
        result = self.align_with_metric(av, metric_depths, metric_skys,
                                        apply_metric_scaling_step=False)

        # 4. Crop padded outputs back to the tile region; un-pad intrinsics.
        #    With MoGe intrinsics the K is built on the tile grid directly
        #    (no pad to un-shift), so it replaces the cropped any-view K.
        result = self._crop_result(result, metas)
        if tile_intrinsics is not None:
            result["intrinsics"] = np.stack(tile_intrinsics)
        return result

    def _run_metric_branch(
        self,
        imgs: list,
        metas: list[dict],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray] | None]:
        """Per-view MoGe v3 inference -> metric depth on the any-view grid.

        Each view is decoded and fed to MoGe at its **original** resolution
        (the fixed 640x480 graph stretches it internally), the metric depth
        is resized back to the view's original pixels and then
        letterbox-padded onto the any-view grid (``meta`` tile scale + pad),
        so the alignment sees matching pixels.  Pad regions and MoGe-masked
        pixels are zeroed (excluded by the alignment's
        ``metric_depth > 1e-2`` mask).

        Returns ``(1, N, H_av, W_av)`` metric depth, the all-zero sky, and —
        only when ``intrinsics_source == "moge3"`` — the per-view MoGe
        intrinsics converted to the output tile grid (``None`` otherwise).
        """
        h, w = self.av.target_h, self.av.target_w
        n = len(imgs)
        grid = np.zeros((1, n, h, w), dtype=np.float32)
        tile_intrinsics: list[np.ndarray] | None = [] if self.intrinsics_source == "moge3" else None
        for i in range(n):
            # Decode to the original-resolution CHW uint8 tensor: MoGe's
            # preprocess stretches it to the fixed graph size internally, and
            # the stretch must be inverted back to the original pixels before
            # placement (mirror of Any2Full's unresize-to-input).
            img = self._load_image(imgs[i])
            out = self.moge3.infer(img)
            # MoGe depth at the fixed graph size; masked-out pixels come back
            # as +inf and are blanked to 0 (excluded from the alignment).
            depth = out["depth"][0].cpu().numpy().astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            if (img.shape[1], img.shape[2]) != depth.shape:
                depth = cv2.resize(depth, (img.shape[2], img.shape[1]),
                                   interpolation=cv2.INTER_LINEAR)
            grid[0, i] = self._place_on_grid(depth, metas[i], h, w)
            if tile_intrinsics is not None:
                tile_intrinsics.append(self._moge3_intrinsics_on_tile(out, metas[i]))
        skys = np.zeros_like(grid)
        return grid, skys, tile_intrinsics

    @staticmethod
    def _moge3_intrinsics_on_tile(out: dict, meta: dict) -> np.ndarray:
        """MoGe per-view intrinsics on the output tile grid, (3, 3) float32.

        MoGe's intrinsics live in its normalized-uv convention (focal in
        image-extent units, principal point at 0.5, pixel centers at
        ``(j + 0.5) / W``).  Converting to pixels exactly like
        ``tools/general_test/module/infer_moge3.py`` does for the 640x480
        grid, but on the any-view **tile** grid the output depth is cropped
        to: focal ``K00/11 * tile_w/h``, principal point
        ``K02/12 * tile_w/h - 0.5`` (each axis follows the stretch of the
        original view onto that tile, so no pad shift applies).
        """
        k = out["intrinsics"][0].cpu().numpy()
        tw, th = float(meta["tile_w"]), float(meta["tile_h"])
        fx, cx = float(k[0, 0]) * tw, float(k[0, 2]) * tw - 0.5
        fy, cy = float(k[1, 1]) * th, float(k[1, 2]) * th - 0.5
        return np.array(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

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
