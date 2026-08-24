# -*- coding: utf-8 -*-
"""
Any2Full_PT2 — torch.export (PT2) runtime for Any2Full.

Loads an exported program (produced by export_any2full_pt2.py), and wraps it
with preprocessing and post-processing:

  preprocess(rgb, depth) -> (rgb, dep)   fixed (1, 3, 480, 640) / (1, 1, 480, 640)
  infer(rgb, dep)        -> (depth, disparity_pre)
  postprocess(depth, disparity_pre, dep) -> final metric depth  (init_scaling affine fit)

The exported graph runs with init_scaling disabled; the scale recovery
(affine least-squares fit of predicted disparity against the sparse anchors)
is re-implemented here as post-processing, deterministically (no jitter).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

from .utils import remove_outliers
from utils.image_io import ImageInput, to_image_tensor


def _load_rgb(rgb: ImageInput) -> torch.Tensor:
    """Convert any ImageInput to a float CHW [0, 1] tensor."""
    rgb = to_image_tensor(rgb)                     # uint8 CHW [0, 255] (or float CHW)
    if rgb.dtype == torch.uint8:
        rgb = rgb.float() / 255.0
    rgb = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(rgb)
    return rgb.unsqueeze(0)


class Any2Full_PT2(nn.Module):
    def __init__(self, pt2_path: str, device="cuda", init_scaling=True,
                 max_depth=1e3, min_depth=1e-6):
        super().__init__()
        exported = torch.export.load(pt2_path)
        self.model = exported.module()
        self.device = device
        self.do_init_scaling = init_scaling
        self.max_depth = max_depth
        self.min_depth = min_depth
        # Precision of the exported graph (fp32 or bf16): inputs must match it.
        params = list(self.model.parameters())
        self.dtype = params[0].dtype if params else torch.float32

    # ---- preprocessing -----------------------------------------------------

    @staticmethod
    def _check_and_resize(x: torch.Tensor, expected_hw, mode: str, name: str) -> torch.Tensor:
        _, _, h, w = x.shape
        exp_h, exp_w = expected_hw
        if (h, w) != (exp_h, exp_w):
            x = F.interpolate(x, size=(exp_h, exp_w), mode=mode, align_corners=None)
        return x

    def preprocess(self, rgb_path: str, depth_metrics: np.ndarray,
                   denoise: bool = False, denoise_kwargs=None) -> tuple:
        rgb = _load_rgb(rgb_path).to(device=self.device, dtype=self.dtype)
        if denoise:
            depth_metrics = remove_outliers(depth_metrics, **(denoise_kwargs or {}))
        # Input sanity check (mirrors the former in-graph assert in
        # Any2Full.forward, which was removed so forward stays exportable).
        if not np.isfinite(depth_metrics).all():
            raise ValueError(f"Input depth contains nan/inf")
        rgb = self._check_and_resize(rgb, (480, 640), "bilinear", "rgb")
        dep = torch.from_numpy(depth_metrics).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        dep = self._check_and_resize(dep, (480, 640), "nearest", "depth")
        return rgb, dep

    # ---- inference ---------------------------------------------------------

    def infer(self, rgb: torch.Tensor, dep: torch.Tensor) -> tuple:
        depth, disparity_pre_internal, prompt_depth_resized = self.model(rgb, dep)
        return depth, disparity_pre_internal, prompt_depth_resized

    # ---- postprocessing ----------------------------------------------------

    @staticmethod
    def disparity_to_depth(disparity: torch.Tensor) -> torch.Tensor:
        disparity = torch.clamp(disparity, min=0)
        eps = 1e-8
        return torch.where(disparity > 0, 1.0 / (disparity + eps), torch.zeros_like(disparity))

    def init_scaling(self, pred: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
        """Affine least-squares fit of predicted disparity against sparse anchors.

        Mirrors Any2Full.init_scaling without the random jitter (deterministic).
        """
        pred = pred.clone().detach()
        for i in range(pred.shape[0]):
            # NOTE: target = sparse[i] (raw), mirroring Any2Full.init_scaling,
            # where `sparse_depth = disparity_to_depth(sparse)` is computed but
            # never used — B comes from the raw sparse disparity values.
            target = sparse[i]
            idx_nnz = torch.nonzero(target.view(-1) > 0.00001, as_tuple=False)
            if idx_nnz.shape[0] == 0:
                continue
            B = target.view(-1)[idx_nnz]
            A = pred[i].view(-1)[idx_nnz]
            num_dep = A.shape[0]
            A = torch.cat((A, torch.ones(num_dep, 1).to(A)), dim=1)
            X = torch.pinverse(A) @ B
            X = X.to(pred)
            pred[i] = pred[i] * X[0] + X[1]
        return pred

    def postprocess(self, depth: torch.Tensor, disparity_pre_internal: torch.Tensor,
                    prompt_depth_resized: torch.Tensor) -> torch.Tensor:
        """Final metric depth.

        With init_scaling, mirrors Any2Full.forward exactly: fit the predicted
        disparity (internal resolution) against the resized sparse anchors,
        convert to depth, unresize to the input resolution, then clamp.
        """
        # Do the scale fit in fp32 regardless of the graph's precision (bf16
        # pinverse would lose too much precision); final output is fp32.
        disparity_pre_internal = disparity_pre_internal.float()
        prompt_depth_resized = prompt_depth_resized.float()
        if self.do_init_scaling:
            target_hw = (depth.shape[-2], depth.shape[-1])  # input-resolution graph depth
            depth = self.disparity_to_depth(
                torch.clamp(self.init_scaling(disparity_pre_internal,
                                               self.disparity_to_depth(prompt_depth_resized)),
                            min=1 / self.max_depth))
            if disparity_pre_internal.shape[-2:] != target_hw:
                depth = F.interpolate(depth, size=target_hw, mode="bilinear", align_corners=True)
        return torch.clamp(depth.float(), min=self.min_depth, max=self.max_depth)



