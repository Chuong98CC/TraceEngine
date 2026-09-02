"""WAFT optical flow inference via a torch.export (``.pt2``) artifact in bf16.

:class:`WAFTv2_PT2` plays the same role as ``model.waft_onnx.WAFTOnnx``
(which pairs ``WAFTBase`` with ``ONNXModel``): it loads the serialized
backend artifact, exposes ``preprocess`` / ``run`` / ``postprocess``, and a
convenience ``__call__(img1, img2)`` returning the flow at the original
resolution.

The artifact is a static-shape ExportedProgram: images must be resized /
padded to the exported ``(target_h, target_w)`` geometry before the run.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import cv2

from .utils.img_utils import bgr_to_rgb, letterbox, load_image, to_model_input

_DTYPE = torch.bfloat16  # the pt2 artifacts are exported in bf16

def postprocess_flow(flow: torch.Tensor, meta: dict) -> np.ndarray:
    """Crop padding and rescale a raw model output to the original size.

    Args:
        flow: Model output ``[1, 2, H, W]`` (bf16, any device) at the
            letterboxed ``(target_h, target_w)`` resolution.
        meta: Letterbox metadata from
            :func:`infer_pt2.utils.img_utils.letterbox` (of image 1).

    Returns:
        np.ndarray: Flow of shape ``(orig_h, orig_w, 2)``, float32 (values
        are in pixels at the original resolution).
    """
    tile_h = meta["tile_h"]
    tile_w = meta["tile_w"]
    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]

    # Crop the letterbox padding (region that carries no image content).
    flow = flow[:, :, pad_top:pad_top + tile_h, pad_left:pad_left + tile_w]

    # Drop batch dim → [tile_h, tile_w, 2] float32 on CPU.
    flow = flow[0].float().cpu().numpy().transpose(1, 2, 0)

    # Resize to the original resolution.
    flow = cv2.resize(
        flow, (meta["orig_w"], meta["orig_h"]), interpolation=cv2.INTER_LINEAR
    )

    # Rescale flow values (flow is in pixels).
    flow[:, :, 0] *= meta["orig_w"] / tile_w
    flow[:, :, 1] *= meta["orig_h"] / tile_h

    return flow
class WAFTv2_PT2:
    """WAFT optical flow inference backed by a torch.export ``.pt2`` artifact.

    Parameters
    ----------
    pt2_path : str
        Path to the ``.pt2`` artifact (see ``infer_pt2/export_pt2.py``).
    device : str
        ``"cuda"`` or ``"cpu"``.  The artifact stores bf16 weights, which are
        moved to this device at load time.
    bgr_input : bool
        If ``True`` (default), input images are BGR (OpenCV convention) and
        are converted to RGB internally.
    target_h, target_w : int, optional
        Override the model input geometry.  If ``None`` (default) the size
        is read from the exported graph placeholders.
    """

    def __init__(
        self,
        pt2_path: str,
        device: str = "cuda",
        bgr_input: bool = True,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> None:
        if not os.path.isfile(pt2_path):
            raise FileNotFoundError(f"PT2 artifact not found: {pt2_path}")

        self.pt2_path = pt2_path
        self.device = device
        self._bgr_input = bgr_input
        self._meta: dict | None = None  # set by preprocess

        # ── Load artifact & move weights to the target device ─────────────
        print(f"Loading PT2 artifact from: {pt2_path} ...")
        ep = torch.export.load(pt2_path)
        self._input_names = list(ep.graph_signature.user_inputs)
        if len(self._input_names) != 2:
            raise RuntimeError(
                f"Expected 2 graph inputs, found {self._input_names}. "
                "Was this artifact exported from WAFTv2?"
            )

        if target_h is None or target_w is None:
            in_h, in_w, in_dtype = self._read_input_size(ep)
            target_h = in_h if target_h is None else target_h
            target_w = in_w if target_w is None else target_w
            if in_dtype != _DTYPE:
                print(
                    f"WARNING: artifact inputs are {in_dtype}, not {_DTYPE} — "
                    "feed tensors with the matching dtype."
                )
        else:
            in_dtype = _DTYPE
        self.target_h = target_h
        self.target_w = target_w

        self._graph_module = ep.module().to(device)  # inference graph: no train/eval modes
        print(
            f"  Input geometry : {self.target_h} x {self.target_w} "
            f"({in_dtype}) on {device}"
        )

    # ------------------------------------------------------------------
    # Artifact introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _read_input_size(ep) -> tuple[int, int, torch.dtype]:
        """Read (H, W, dtype) of the first user input from the exported graph.

        ``torch.export`` emits placeholder nodes for the model inputs; their
        ``meta["val"]`` holds the static shapes baked in at export time.
        """
        user_inputs = set(ep.graph_signature.user_inputs)
        for node in ep.graph.nodes:
            if node.op != "placeholder" or node.name not in user_inputs:
                continue
            val = node.meta.get("val")
            if val is None or tuple(val.shape)[0] != 1:
                continue
            shape = tuple(val.shape)
            if len(shape) == 4:  # [1, 3, H, W]
                return shape[2], shape[3], val.dtype
        raise RuntimeError(
            "Could not determine the input size from the exported graph. "
            "Pass target_h / target_w explicitly."
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(self, img1, img2) -> dict[str, torch.Tensor]:
        """Preprocess a pair of images into a feed dict for the artifact.

        Parameters
        ----------
        img1, img2 : str or np.ndarray
            Path to the image, or uint8 H×W×3 array.

        Returns
        -------
        dict
            ``{"image1": (1,3,H,W) bf16, "image2": (1,3,H,W) bf16}`` on
            ``self.device`` with values in [0, 255] (the exported model
            normalizes internally).  Also stores letterbox metadata in
            ``self._meta`` for :meth:`postprocess`.
        """
        img1 = load_image(img1)
        img2 = load_image(img2)

        if self._bgr_input:
            img1 = bgr_to_rgb(img1)
            img2 = bgr_to_rgb(img2)

        # Letterbox (metadata from image1 drives both for consistency)
        img1_padded, meta = letterbox(img1, self.target_h, self.target_w)
        img2_padded, _ = letterbox(img2, self.target_h, self.target_w)
        self._meta = meta

        return {
            "image1": to_model_input(img1_padded, _DTYPE, self.device),
            "image2": to_model_input(img2_padded, _DTYPE, self.device),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, feed: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the exported graph on a preprocessed feed dict.

        Parameters
        ----------
        feed : dict
            Output of :meth:`preprocess` (two bf16 tensors on ``self.device``).

        Returns
        -------
        dict
            ``{"flow": [1, 2, H, W] bf16}`` on ``self.device``.
        """
        inputs = [feed[name] for name in self._input_names]
        with torch.no_grad():
            flow = self._graph_module(*inputs)
        return {"flow": flow}

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def postprocess(self, raw_output: dict[str, torch.Tensor]) -> np.ndarray:
        """Crop padding and rescale flow back to the original resolution.

        Parameters
        ----------
        raw_output : dict
            Output of :meth:`run`, must contain ``"flow"`` → ``[1, 2, H, W]``.

        Returns
        -------
        np.ndarray
            Flow array of shape ``(orig_h, orig_w, 2)``, float32.
        """
        if self._meta is None:
            raise RuntimeError(
                "No preprocessing metadata; call preprocess() before postprocess()."
            )
        return postprocess_flow(raw_output["flow"], self._meta)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __call__(self, img1, img2) -> np.ndarray:
        """Run the full pipeline: preprocess → infer → postprocess.

        Parameters
        ----------
        img1, img2 : str or np.ndarray
            Image paths or uint8 H×W×3 arrays.

        Returns
        -------
        np.ndarray
            Optical flow ``(orig_h, orig_w, 2)``, float32.
        """
        feed = self.preprocess(img1, img2)
        raw = self.run(feed)
        return self.postprocess(raw)
