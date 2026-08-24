"""Backend-agnostic Depth Anything 3 pre/post-processing mixin.

Holds the preprocessing, extrinsics normalization, output-name mapping, and
alignment orchestration shared by every DA3 inference wrapper.  It is a pure
mixin: it defines no ``__init__`` and relies on ``self.target_h`` / ``self.target_w``
supplied by the backend base (``ONNXModel`` now; ``TRTModel`` later).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from depth_models.da3.utils.alignment import (align_anyview_with_metric,
                                   align_to_input_ext_scale)


class BaseDA3Model:
    """Shared DA3 preprocessing + alignment.  Requires ``self.target_h/target_w``."""

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ---- preprocessing (letterbox: aspect-preserve + pad) -------------------

    def preprocess_views(
        self,
        imgs: list,
        intrs: np.ndarray,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Resize/normalize *N* views to ``(1, N, 3, H, W)`` with scaled intrinsics.

        ``target_h``/``target_w`` default to ``self.target_h``/``self.target_w``;
        pass explicit values to preprocess for a differently-sized model (e.g. the
        metric branch inside the nested pipeline).
        """
        th = target_h if target_h is not None else self.target_h
        tw = target_w if target_w is not None else self.target_w
        n = len(imgs)
        proc = np.zeros((n, 3, th, tw), dtype=np.float32)
        intrs_out = np.zeros((n, 3, 3), dtype=np.float32)
        metas: list[dict] = []
        for i in range(n):
            proc[i], intrs_out[i], meta = self._preprocess_one(imgs[i], intrs[i], th, tw)
            metas.append(meta)
        return proc[None], intrs_out, metas  # add B=1

    def _preprocess_one(
        self,
        img,
        K: np.ndarray,
        target_h: int,
        target_w: int,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Letterbox a single view (path or BGR array): aspect-preserving resize
        with a 2-decimal-truncated scale, then center-pad to the target — mirrors
        ``TRTModel.resize_img``.  Returns CHW float32, pad/scale-adjusted intrinsics,
        and per-view meta for the crop-back on post-process.
        """
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

        # Uniform scale + pad offset on the principal point.
        K_adj = K.copy().astype(np.float32)
        K_adj[0, 0] *= scale
        K_adj[1, 1] *= scale
        K_adj[0, 2] = K[0, 2] * scale + pad_left
        K_adj[1, 2] = K[1, 2] * scale + pad_top

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return chw, K_adj, meta

    @staticmethod
    def crop_to_tile(arr: np.ndarray, meta: dict) -> np.ndarray:
        """Crop the last two dims to the unpadded tile region (mirrors
        ``DA3MetricTRT.parse_outputs``' crop)."""
        pt, pl = meta["pad_top"], meta["pad_left"]
        th, tw = meta["tile_h"], meta["tile_w"]
        return arr[..., pt : pt + th, pl : pl + tw]

    # ---- extrinsics normalization ------------------------------------------

    @staticmethod
    def normalize_extrinsics(extrs: np.ndarray) -> np.ndarray:
        """First camera to origin, median camera distance = 1 (clamped 1e-1).

        Numpy replica of ``DepthAnything3._normalize_extrinsics``.  ``extrs`` is
        ``(N, 4, 4)`` world-to-camera.
        """
        ex_t = extrs.copy()
        transform = np.linalg.inv(ex_t[0])
        ex_t_norm = ex_t @ transform
        c2ws = np.linalg.inv(ex_t_norm)
        dists = np.linalg.norm(c2ws[..., :3, 3], axis=-1)
        median_dist = max(float(np.median(dists)), 1e-1)
        ex_t_norm[..., :3, 3] /= median_dist
        return ex_t_norm

    # ---- any-view input assembly -------------------------------------------

    @property
    def uses_extrinsics(self) -> bool:
        """True if the loaded graph declares an ``extrinsics`` input.

        Distinguishes the "with-camera-pose" any-view export (inputs
        ``image``/``extrinsics``/``intrinsics``) from the default export
        (``image`` only, the model predicts its own poses).  Reads the backend's
        ``self.inputs`` metadata so feeding is driven by the actual model, never a
        caller-supplied flag.
        """
        return any(i["name"] == "extrinsics" for i in getattr(self, "inputs", []))

    def build_anyview_feed(
        self,
        img_batch: np.ndarray,
        extrs: np.ndarray | None,
        intrs_adj: np.ndarray,
        *,
        normalize_extrinsics: bool = False,
    ) -> dict[str, np.ndarray]:
        """Assemble the any-view input feed for the loaded graph.

        Always includes ``image``; adds ``extrinsics``/``intrinsics`` only when the
        graph declares them (see :attr:`uses_extrinsics`).  ``extrs`` may be
        ``None`` for a default (pose-predicting) export.  Extrinsics are normalized
        here only when ``normalize_extrinsics=True`` (the graph does not normalize).
        """
        feed: dict[str, np.ndarray] = {"image": img_batch.astype(np.float32)}
        if self.uses_extrinsics:
            if extrs is None:
                raise ValueError(
                    "This any-view model was exported with --use-extrinsics and "
                    "requires camera extrinsics/intrinsics, but none were provided."
                )
            ext = self.normalize_extrinsics(extrs) if normalize_extrinsics else extrs
            feed["extrinsics"] = ext[None].astype(np.float32)
            feed["intrinsics"] = intrs_adj[None].astype(np.float32)
        return feed

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
    def apply_mono_sky(
        depth: np.ndarray, sky: np.ndarray, threshold: float = 0.3
    ) -> np.ndarray:
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
    ) -> dict[str, np.ndarray]:
        """Run ``align_anyview_with_metric`` and squeeze the batch dim.

        ``av`` holds numpy any-view outputs (batched, ``(1,N,…)``).
        ``metric_depths``/``metric_skys`` are ``(1, N, H, W)``.  Returns numpy
        ``{depth, depth_conf, extrinsics, intrinsics}`` squeezed to ``(N, …)``.
        """
        aligned = align_anyview_with_metric(
            anyview_depth=torch.from_numpy(av["depth"]),
            anyview_conf=torch.from_numpy(av["depth_conf"]),
            anyview_extrinsics=torch.from_numpy(av["extrinsics"]),
            anyview_intrinsics=torch.from_numpy(av["intrinsics"]),
            metric_depth=torch.from_numpy(metric_depths),
            metric_sky=torch.from_numpy(metric_skys),
        )
        out: dict[str, np.ndarray] = {}
        for k in ("depth", "depth_conf", "extrinsics", "intrinsics"):
            val = aligned[k].float().cpu().numpy()
            if val.ndim >= 4 and val.shape[0] == 1:
                val = val.squeeze(0)
            out[k] = val
        return out

    def align_to_input(
        self,
        result: dict,
        input_extrinsics: np.ndarray,
        input_intrinsics: np.ndarray,
        align_scale: bool = True,
    ) -> dict:
        """Umeyama-align the prediction to the input camera poses (in place-ish).

        ``align_scale`` mirrors ``DepthAnything3.inference``'s
        ``align_to_input_ext_scale``:

        - ``True`` (default): replace the output poses with the raw input poses and
          divide depth by the Umeyama scale (output is in the input-pose scale).
        - ``False``: keep the **predicted** poses, only rigidly aligned into the
          input frame (Umeyama rotation + translation), and leave depth unchanged.
        """
        aligned = align_to_input_ext_scale(
            pred_depth=result["depth"],
            pred_extrinsics=result["extrinsics"],
            input_extrinsics=input_extrinsics,
            input_intrinsics=input_intrinsics,
            align_scale=align_scale,
        )
        result = dict(result)
        result["depth"] = aligned["depth"]
        result["extrinsics"] = aligned["extrinsics"]
        result["intrinsics"] = aligned["intrinsics"]
        return result
