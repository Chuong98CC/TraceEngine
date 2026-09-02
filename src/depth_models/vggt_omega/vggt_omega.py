"""VGGT-Omega inference wrapper for torch.export programs.

``BaseVGGTOmega`` holds the backend-agnostic pre/post (letterbox, feed
assembly, pose/depth post-processing); the concrete wrapper supplies the
static input geometry, the name-keyed ``self.inputs`` metadata, and
``_forward``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils.file_io.image_io import ImageInput

from .base_vggt_omega import BaseVGGTOmega


def move_lifted_state_to_device(exported: "torch.export.ExportedProgram", device: str | torch.device) -> None:
    """Move an ExportedProgram's lifted params/buffers/constants to ``device``.

    ``torch.export.load`` materializes everything on cpu; the exported graph's
    runtime inputs may live on cuda, so the lifted tensors must be moved before
    ``exported.module()`` is built (``.to()`` on the built module misses the
    non-persistent-buffer constants).
    """
    for name in list(exported.state_dict):
        value = exported._state_dict[name]
        moved = value.to(device)
        exported._state_dict[name] = (
            torch.nn.Parameter(moved) if isinstance(value, torch.nn.Parameter) else moved
        )
    for name in list(exported.constants):
        value = exported.constants[name]
        if isinstance(value, torch.Tensor):
            exported.constants[name] = value.to(device)


class VGGT_Omega(BaseVGGTOmega):
    """Run an exported VGGT-Omega model end-to-end.

    Despite the name, the file is not a TorchScript (jit) module — it is an
    ExportedProgram saved by torch.export, in either the classic pickle .pt2
    or the newer zip-based .pt format. The static (N, H, W) input size is
    read from the saved example input; ``infer`` takes N images in any
    ``ImageInput`` form — paths, RGB arrays or CHW tensors — decoded and
    letterboxed by ``BaseVGGTOmega`` (via the shared ``utils.image_io``
    helpers) into a CPU float32 ``(1, N, 3, H, W)`` feed that ``_forward``
    casts to the export's dtype and runs on cuda.  Spatial outputs are
    cropped back to the content region.  Mean/std normalization happens
    inside the model, so the feed only needs to be float32 RGB in [0,1]
    (done by ``BaseVGGTOmega``).

    Runs on CUDA: the exported graph is device-baked at trace time (RoPE
    derives its arange device from the lifted weights), and the lifted
    params/constants are moved to cuda on load.
    """

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            self._exported = torch.export.load(f)
        args = self._exported.example_inputs[0]
        example = args[0] if isinstance(args, (tuple, list)) else args
        shape = tuple(example.shape)
        if len(shape) != 5 or shape[0] != 1:
            raise ValueError(
                f"Model input must be (1, N, 3, H, W) with a static N; "
                f"got example input shape {shape}"
            )
        self.num_views = shape[1]
        self.target_h, self.target_w = shape[3], shape[4]
        self._input_fp16 = example.dtype == torch.float16
        self.inputs = [
            {
                "name": "images",
                "shape": shape,
                "dtype": np.float16 if self._input_fp16 else np.float32,
            }
        ]
        move_lifted_state_to_device(self._exported, "cuda")
        self._graph_module = self._exported.module()

    def infer(self, images: list[ImageInput]) -> dict[str, np.ndarray]:
        """Run inference on exactly N image paths, RGB arrays or tensors.

        N is the static frame count the model was exported with; see
        ``BaseVGGTOmega.infer`` for the pre/post-processing pipeline.
        """
        if len(images) != self.num_views:
            raise ValueError(
                f"This model is static for N={self.num_views} views; "
                f"got {len(images)} images"
            )
        return super().infer(images)

    # Backward-compatible alias for the standalone ``run`` API.
    run = infer

    def _forward(self, feed: dict) -> dict[str, np.ndarray]:
        """Run the exported graph on the ``(1, N, 3, H, W)`` CPU tensor feed,
        casting to the export's input dtype; returns the raw ``(1, N, ...)``
        outputs keyed for ``BaseVGGTOmega.postprocess``."""
        batch = next(iter(feed.values()))
        dtype = torch.float16 if self._input_fp16 else torch.float32
        with torch.inference_mode():
            pose_e, depth_e, conf_e = self._graph_module(
                batch.to(dtype=dtype, device="cuda")
            )
        return {
            "pose_enc": pose_e.float().cpu().numpy(),
            "depth": depth_e.float().cpu().numpy(),
            "depth_conf": conf_e.float().cpu().numpy(),
        }
