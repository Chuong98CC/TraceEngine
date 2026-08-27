"""Backend-agnostic Depth Anything 3 pre/post-processing mixin.

Holds the letterbox preprocessing, output-name mapping, and alignment
orchestration shared by the torch.export wrappers.  It is a pure mixin: it
defines no ``__init__`` and relies on ``self.target_h`` / ``self.target_w``
supplied by the backend (``DA3AnyViewPT2`` / ``DA3MetricPT2``).  Standalone
copy of ``tools/model/base_da3.py``, with the input-camera (extrinsics /
intrinsics) paths removed — the exported graphs predict their own poses and
intrinsics, so no camera parameters are ever consumed (``cv2`` is imported
lazily in ``_preprocess_one`` so the package also imports without OpenCV).
"""

from __future__ import annotations

import numpy as np
import torch
from depth_models.da3.utils.alignment import align_anyview_with_metric


class BaseDA3Model:
    """Shared DA3 preprocessing + alignment.  Requires ``self.target_h/target_w``."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ---- preprocessing (letterbox: aspect-preserve + pad) -------------------

    def preprocess_views(
        self,
        imgs: list,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Resize/normalize *N* views to ``(1, N, 3, H, W)``.

        ``target_h``/``target_w`` default to ``self.target_h``/``self.target_w``;
        pass explicit values to preprocess for a differently-sized model (e.g. the
        metric branch inside the nested pipeline).  Returns the padded batch and
        per-view crop-back metadata.
        """
        th = target_h if target_h is not None else self.target_h
        tw = target_w if target_w is not None else self.target_w
        n = len(imgs)
        proc = np.zeros((n, 3, th, tw), dtype=np.float32)
        metas: list[dict] = []
        for i in range(n):
            proc[i], meta = self._preprocess_one(imgs[i], th, tw)
            metas.append(meta)
        return proc[None], metas  # add B=1

    def _preprocess_one(
        self,
        img,
        target_h: int,
        target_w: int,
    ) -> tuple[np.ndarray, dict]:
        """Letterbox a single view (path or BGR array): aspect-preserving resize
        with a 2-decimal-truncated scale, then center-pad to the target.
        Returns CHW float32 and per-view meta for the crop-back on post-process.
        """
        import cv2  # noqa: PLC0415  (lazy so the package imports without OpenCV)

        if isinstance(img, str):
            bgr = cv2.imread(img, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Could not load image: {img}")
        elif isinstance(img, np.ndarray):
            bgr = img.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        orig_h, orig_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Uniform scale, truncated to 2 decimals so the leftover fit is padded.
        raw_scale = min(target_w / orig_w, target_h / orig_h)
        scale = np.floor(raw_scale * 100.0) / 100.0
        if scale <= 0:
            scale = raw_scale

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        img_f = padded.astype(np.float32) / 255.0
        chw = ((img_f - self._MEAN) / self._STD).transpose(2, 0, 1).astype(np.float32)

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return chw, meta

    @staticmethod
    def crop_to_tile(arr: np.ndarray, meta: dict) -> np.ndarray:
        """Crop the last two dims to the unpadded tile region (mirrors
        ``DA3MetricTRT.parse_outputs``' crop)."""
        pt, pl = meta["pad_top"], meta["pad_left"]
        th, tw = meta["tile_h"], meta["tile_w"]
        return arr[..., pt : pt + th, pl : pl + tw]

    # ---- output-name resolvers ---------------------------------------------

    @staticmethod
    def map_anyview_keys(raw: dict) -> dict:
        """Normalise any-view output keys to ``depth/depth_conf/extrinsics/intrinsics``."""
        out: dict[str, np.ndarray] = {}
        for name, val in raw.items():
            low = name.lower()
            if "depth_conf" in low or "conf" in low:
                out["depth_conf"] = val
            elif "pred_extrinsics" in low:
                out["extrinsics"] = val
            elif "pred_intrinsics" in low:
                out["intrinsics"] = val
            elif "depth" in low:
                out["depth"] = val
            else:
                out[name] = val
        return out

    @staticmethod
    def extract_metric(raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract ``(depth, sky)`` from raw metric outputs."""
        depth = sky = None
        for name, val in raw.items():
            low = name.lower()
            arr = val.squeeze().astype(np.float32)
            if "sky" in low:
                sky = arr
            elif "depth" in low:
                depth = arr
            elif depth is None:
                depth = arr
        if sky is None:
            sky = np.zeros_like(depth)
        return depth, sky

    @staticmethod
    def apply_mono_sky(depth: np.ndarray, sky: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Clamp sky-region depth to the non-sky 99th percentile (mono-sky post-proc).

        NumPy replica of ``DepthAnything3Net._process_mono_sky_estimation`` — the
        step the metric ONNX/TRT graph omits (its ``torch.quantile`` lowers to an
        ONNX ``TopK`` that exceeds TensorRT limits, so it runs here, outside the
        graph).  Applies only when both classes have >10 pixels; deterministic
        (first 100k non-sky samples), matching the model's export-friendly forward.

        Must run on the padded model-resolution output **before** any crop, so the
        sample set and mask match the PyTorch forward.  Standalone metric inference
        only — the nested pipeline feeds the raw metric depth to alignment.
        """
        non_sky = sky < threshold  # True = non-sky
        if int(non_sky.sum()) <= 10 or int((~non_sky).sum()) <= 10:
            return depth
        sampled = depth[non_sky].reshape(-1)[:100_000]
        # Append a single zero to guard the quantile against an empty tensor,
        # exactly as the traced PyTorch path does.
        safe = np.concatenate([sampled, np.zeros(1, dtype=sampled.dtype)])
        non_sky_max = np.quantile(safe, 0.99)
        out = depth.copy()
        out[~non_sky] = non_sky_max
        return out

    # ---- alignment orchestration -------------------------------------------

    def align_with_metric(
        self,
        av: dict,
        metric_depths: np.ndarray,
        metric_skys: np.ndarray,
        apply_metric_scaling_step: bool = True,
    ) -> dict[str, np.ndarray]:
        """Run ``align_anyview_with_metric`` and squeeze the batch dim.

        ``av`` holds numpy any-view outputs (batched, ``(1,N,…)``).
        ``metric_depths``/``metric_skys`` are ``(1, N, H, W)``.  Returns numpy
        ``{depth, depth_conf, extrinsics, intrinsics}`` squeezed to ``(N, …)``.
        ``apply_metric_scaling_step=False`` skips the focal rescale for metric
        depths that are already metric metres (Any2Full).
        """
        aligned = align_anyview_with_metric(
            anyview_depth=torch.from_numpy(av["depth"]),
            anyview_conf=torch.from_numpy(av["depth_conf"]),
            anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
            anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
            metric_depth=torch.from_numpy(metric_depths),
            metric_sky=torch.from_numpy(metric_skys),
            apply_metric_scaling_step=apply_metric_scaling_step,
        )
        out: dict[str, np.ndarray] = {}
        for k in ("depth", "depth_conf", "extrinsics", "intrinsics"):
            val = aligned[k].float().cpu().numpy()
            if val.ndim >= 4 and val.shape[0] == 1:
                val = val.squeeze(0)
            out[k] = val
        return out
