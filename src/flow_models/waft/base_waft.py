"""Backend-agnostic pre/post-processing for WAFT optical flow inference.

``WAFTBase`` owns the image-level pipeline (load, BGR→RGB, letterbox,
post-inference flow crop/rescale).  Tensor format conversion
(HWC→NCHW float32) and inference dispatch are the responsibility of each
concrete backend subclass — see :class:`WAFTOnnx` and :class:`WAFTRT`.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


class WAFTBase:
    """Backend-agnostic preprocessing and postprocessing for WAFT flow models.

    Parameters
    ----------
    bgr_input : bool
        If ``True`` (default), input images are assumed to be BGR (the
        OpenCV convention) and are converted to RGB before inference.
    """

    def __init__(self, bgr_input: bool = True) -> None:
        self._bgr_input = bgr_input
        self._meta: dict | None = None  # set by preprocess

    # ------------------------------------------------------------------
    # Preprocessing  (image-level only — no tensor format)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(img) -> np.ndarray:
        """Return a uint8 H×W×3 image from a path or an existing array."""
        if isinstance(img, str):
            if not os.path.isfile(img):
                raise FileNotFoundError(f"Image not found: {img}")
            arr = cv2.imread(img)
            if arr is None:
                raise FileNotFoundError(
                    f"Could not read image (check format / permissions): {img}"
                )
            return arr
        if isinstance(img, np.ndarray):
            return img
        raise TypeError(
            f"Expected str (path) or np.ndarray, got {type(img).__name__}"
        )

    def _letterbox(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        """Aspect-preserving resize + centre-pad to ``(target_h, target_w)``.

        Returns the padded uint8 HWC image and a metadata dict consumed by
        :meth:`postprocess`.
        """
        orig_h, orig_w = img.shape[:2]

        raw_scale = min(self.target_w / orig_w, self.target_h / orig_h)
        scale_factor = np.floor(raw_scale * 100.0) / 100.0
        if scale_factor <= 0:
            scale_factor = raw_scale

        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        img_resized = cv2.resize(img, (new_w, new_h))

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        img_padded = cv2.copyMakeBorder(
            img_resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "scale_factor": float(scale_factor),
            "tile_h": new_h,
            "tile_w": new_w,
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
        }
        return img_padded, meta

    def preprocess(self, img1, img2) -> tuple[np.ndarray, np.ndarray]:
        """Load, optionally convert colour space, and letterbox an image pair.

        Returns padded **uint8 HWC** images ready for backend-specific
        tensor conversion.  Letterbox metadata is stored in ``self._meta``
        for :meth:`postprocess`.
        """
        img1 = self._load_image(img1)
        img2 = self._load_image(img2)

        if self._bgr_input:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        img1_padded, meta = self._letterbox(img1)
        img2_padded, _ = self._letterbox(img2)
        self._meta = meta

        return img1_padded, img2_padded

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(self, raw_output: dict) -> np.ndarray:
        """Crop padding and rescale flow back to the original resolution.

        Parameters
        ----------
        raw_output : dict
            Backend output dict; must contain ``"flow"`` → ``[1, 2, H, W]``.

        Returns
        -------
        np.ndarray
            Optical flow ``(orig_h, orig_w, 2)``, float32.
        """
        if self._meta is None:
            raise RuntimeError(
                "No preprocessing metadata; call preprocess() before postprocess()."
            )

        meta = self._meta
        flow = raw_output["flow"]  # [1, 2, H, W]

        pt = meta["pad_top"]
        pl = meta["pad_left"]
        flow = flow[:, :, pt:pt + meta["tile_h"], pl:pl + meta["tile_w"]]

        flow = flow[0]  # → [2, tile_h, tile_w]
        flow = flow.transpose(1, 2, 0)  # → [tile_h, tile_w, 2]

        flow = cv2.resize(
            flow, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR
        )

        flow[:, :, 0] *= meta["orig_w"] / meta["tile_w"]
        flow[:, :, 1] *= meta["orig_h"] / meta["tile_h"]

        return flow

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_flow(raw: dict) -> dict:
        """Replace NaN / Inf flow values with zero."""
        flow = raw.get("flow")
        if flow is not None:
            raw["flow"] = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        return raw

    def __call__(self, img1, img2) -> np.ndarray:
        """Run the full pipeline: preprocess → infer → postprocess.

        Subclasses must implement ``run(img1_padded, img2_padded) → dict``
        where both inputs are padded uint8 HWC arrays.
        """
        img1_padded, img2_padded = self.preprocess(img1, img2)
        raw = self.run(img1_padded, img2_padded)  # pylint: disable=no-member
        return self.postprocess(self._sanitize_flow(raw))
