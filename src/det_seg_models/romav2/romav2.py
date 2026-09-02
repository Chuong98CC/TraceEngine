"""Standalone RoMaV2 inference wrapper for an exported .pt2 program.

Only depends on torch/numpy/PIL — no `romav2` package, no checkpoint download.
Image decoding is shared via ``utils.image_io`` (PIL / torchvision).  The
exported program expects fixed-resolution inputs (see config.json written
next to the .pt2); `match_pair` resizes inputs accordingly, mirroring
RoMaV2.match. `match` runs the multi-image strategy on top of `match_pair`.

Usage:
    from romav2_pt2 import RoMaV2PT2

    model = RoMaV2PT2("romav2_pt2.pt2")
    preds = model.match_pair("img_A.png", "img_B.png")
    matches, overlaps, precision_AB, precision_BA = model.sample(preds, 5000)
    kptsA, kptsB = model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .utils import (
    bhwc_grid_sample,
    compute_cycle_errors,
    filter_matches,
    get_normalized_grid,
    kde,
    sample_overlap,
    select_top_k,
    to_pixel,
    warp_points,
)
from utils.file_io.image_io import ImageInput, to_image_tensor, to_pixel_uint8

DEFAULT_MODEL_PATH: str = "weights/romav2/romav2.pt2"

class RoMaV2PT2:
    """Standalone inference wrapper around an exported RoMaV2 .pt2 program."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        model_path = Path(model_path)
        config_path = model_path.with_suffix(".json")
        self.cfg: dict = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.H_lr: int = self.cfg["H_lr"]
        self.W_lr: int = self.cfg["W_lr"]
        # The exported program always computes both directions.
        self.bidirectional: bool = True
        torch.set_float32_matmul_precision("highest")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device: torch.device = torch.device(device)
        # NOTE: the .pt2 stores tensors on the device it was exported from
        # (CUDA here); running on a CPU-only machine requires a CPU export.
        # ExportedProgram.module() is always inference-mode (eval() unsupported).
        # torch.export.load maps the archive's read-only buffers with
        # torch.frombuffer, which warns once per process that the buffer is
        # not writable (a harmless artifact of the zip-backed archive)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message=r"The given buffer is not writable.*")
            self.module: torch.nn.Module = torch.export.load(model_path).module()

    def _load_image(self, img_like: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU).

        numpy arrays must be uint8 RGB (HWC); CHW tensors may be uint8 or
        float [0,1] (rescaled to uint8 pixel space).
        """
        if isinstance(img_like, (str, Path)):
            with Image.open(img_like) as im:
                mode = im.mode
        elif isinstance(img_like, Image.Image):
            mode = img_like.mode
        else:
            mode = None
        if mode == "I;16":
            raise NotImplementedError("Can't handle 16 bit images")
        return to_pixel_uint8(to_image_tensor(img_like))

    @torch.inference_mode()
    def match_pair(
        self, img_A: ImageInput, img_B: ImageInput
    ) -> dict[str, torch.Tensor]:
        img_A = self._load_image(img_A).float().div(255.0).unsqueeze(0).to(self.device)
        img_B = self._load_image(img_B).float().div(255.0).unsqueeze(0).to(self.device)

        img_A = F.interpolate(
            img_A,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        img_B = F.interpolate(
            img_B,
            size=(self.H_lr, self.W_lr),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        (
            warp_AB,
            confidence_AB,
            overlap_AB,
            precision_AB,
            warp_BA,
            confidence_BA,
            overlap_BA,
            precision_BA,
        ) = self.module(img_A, img_B)

        return {
            "warp_AB": warp_AB,
            "confidence_AB": confidence_AB,
            "overlap_AB": overlap_AB,
            "precision_AB": precision_AB,
            "warp_BA": warp_BA,
            "confidence_BA": confidence_BA,
            "overlap_BA": overlap_BA,
            "precision_BA": precision_BA,
        }

    def match(
        self,
        images: list[ImageInput] | torch.Tensor,
        strategy: str = "reference",
        num_corresp: int = 500,
        overlap_th: float = 0.5,
        top_k: int | None = None,
        cycle_th: float = 0.01,
        masks: list[np.ndarray | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Match images[0] against the rest; return surviving tracks.

        Finds points visible in ALL given images. Points are anchored in the
        first image (sampled with balanced sampling from the (0, 1) pair) and
        projected into every other image through that pair's dense warp field.

        images: a list of image paths, PIL images or uint8 RGB (H, W, 3)
        arrays, or a (B, 3, H, W) tensor (uint8 or float in [0, 1]). Each
        element is loaded the same way as in match_pair (see _load_image).

        Strategies:
            reference  -- match image 0 against every other image (N-1 calls);
                          positions come straight from each pair's warp field.
            cycle      -- additionally match all remaining pairs (N(N-1)/2
                          calls total) and keep only points whose tracked
                          positions agree with every pair's warp in both
                          directions (round-trip error below cycle_th).

        With 2 images this reduces to the plain pairwise case: one match_pair
        call, and the overlap filter is the only difference from sample()
        output.

        Selection (choose one): threshold mode (default) keeps every candidate
        whose overlap exceeds overlap_th in ALL images; pass top_k instead to
        keep only the top_k candidates ranked by worst-case overlap across
        images (min over images), ignoring overlap_th. In cycle mode both
        selections first drop round-trip-inconsistent candidates (cycle_th is
        a hard filter, not a ranking criterion). When fewer than top_k
        candidates survive, all of them are kept.

        masks: one (H, W) bool/uint8 object mask per image (at the input-image
        resolution; None entries allowed), passed to sample() so the base
        points on image 0 are sampled inside masks[0] (and the (0, 1) pair's
        candidate pool is gated by masks[0]/masks[1]). Positions in the other
        images are still warp-derived and are not mask-filtered here.

        Returns (positions, mask): positions (K, N, 2) normalized coordinates
        of the surviving points in each image; mask (M,) indexing the sampled
        candidates (candidates: num_corresp at most).
        """
        if isinstance(images, torch.Tensor):
            images = list(images.unbind())  # (B, 3, H, W) -> per-image tensors
        N = len(images)
        if masks is not None and len(masks) != N:
            raise ValueError(
                f"masks must have one entry per image (got {len(masks)}, "
                f"expected {N})"
            )
        if strategy == "cycle":
            pairs = list(combinations(range(N), 2))
        else:
            pairs = [(0, j) for j in range(1, N)]

        preds = {}
        for i, j in pairs:
            preds[(i, j)] = self.match_pair(images[i], images[j])

        # Sample base points on image 0 (balanced sampling from the (0, 1) pair).
        pair_masks = [masks[0], masks[1]] if masks is not None else None
        matches, _, _, _ = self.sample(
            preds[(0, 1)], num_corresp, masks=pair_masks
        )
        base_pts = matches[:, :2]  # (M, 2) normalized

        M = base_pts.shape[0]
        positions = torch.zeros(M, N, 2, device=base_pts.device)
        overlaps = torch.zeros(M, N, device=base_pts.device)
        positions[:, 0] = base_pts
        overlaps[:, 0] = 1.0  # trivially visible in itself
        for j in range(1, N):
            p0j = preds[(0, j)]
            positions[:, j] = warp_points(p0j["warp_AB"], base_pts)
            overlaps[:, j] = sample_overlap(p0j["overlap_AB"], base_pts)

        roundtrip_err = None
        if strategy == "cycle":
            pair_warps = {
                pair: (preds[pair]["warp_AB"], preds[pair]["warp_BA"])
                for pair in pairs
            }
            roundtrip_err = compute_cycle_errors(positions, pair_warps)

        if top_k is not None:
            mask = select_top_k(
                overlaps, top_k, roundtrip_err=roundtrip_err, cycle_th=cycle_th
            )
        else:
            mask = filter_matches(
                overlaps,
                roundtrip_err=roundtrip_err,
                overlap_th=overlap_th,
                cycle_th=cycle_th,
            )
        return positions[mask], mask

    def _mask_flags(
        self,
        masks: list[np.ndarray | None],
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-candidate in-mask flags for both anchor images, (2*H*W,) bool.

        masks: [mask_A, mask_B] — one (H, W) bool/uint8 array per image of the
        pair, at the input-image resolution (any size; nearest-resized to the
        low-res grid), or None to leave that image unconstrained. matches_AB
        row i corresponds to grid pixel i of image A (row-major) and matches_BA
        row i to pixel i of image B, so the flags are the flattened resized
        masks.
        """
        if len(masks) != 2:
            raise ValueError(
                f"masks must have one entry per image of the pair "
                f"(got {len(masks)}, expected 2)"
            )
        flags = []
        for m in masks:
            if m is None:
                flags.append(torch.ones(H * W, dtype=torch.bool, device=device))
                continue
            m = np.asarray(m)
            if m.ndim != 2:
                raise ValueError(f"mask must be (H, W), got {m.shape}")
            m_t = torch.from_numpy((m > 0).astype(np.float32)).to(device)
            m_t = F.interpolate(m_t[None, None], size=(H, W), mode="nearest")
            flags.append(m_t[0, 0].bool().reshape(-1))
        return torch.cat(flags)

    def sample(
        self,
        preds: dict[str, torch.Tensor],
        num_corresp: int,
        masks: list[np.ndarray | None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Sample correspondences from a match_pair prediction.

        masks: [mask_A, mask_B] object masks of the two images (see
        _mask_flags) — candidates are only sampled inside the mask of their
        anchor image; None entries leave an image unconstrained. When a mask
        leaves no valid candidates (or fewer than requested), the returned
        tensors are truncated or empty instead of erroring.
        """
        warp = preds["warp_AB"]
        confidence_AB = preds["overlap_AB"]
        precision_AB = preds["precision_AB"] if "precision_AB" in preds else None

        warp = warp[0]
        confidence_AB = confidence_AB[0].reshape(-1)
        if precision_AB is not None:
            precision_AB = precision_AB[0]

        H_A, W_A, two = warp.shape
        grid = get_normalized_grid(1, H_A, W_A, warp.device)[0]
        matches_AB = torch.cat((grid, warp), dim=-1).reshape(-1, 4)

        confidence_BA = preds["overlap_BA"]
        warp_BA = preds["warp_BA"]
        precision_BA = preds["precision_BA"] if "precision_BA" in preds else None
        warp_BA = warp_BA[0]
        confidence_BA = confidence_BA[0]
        if precision_BA is not None:
            precision_BA = precision_BA[0]

        if precision_BA is not None and precision_AB is not None:
            precision_A = bhwc_grid_sample(
                precision_BA[None].reshape(1, H_A, W_A, -1),
                warp[None],
                mode="bilinear",
                align_corners=False,
            ).reshape(H_A, W_A, 2, 2)
            precision_B = bhwc_grid_sample(
                precision_AB[None].reshape(1, H_A, W_A, -1),
                warp_BA[None],
                mode="bilinear",
                align_corners=False,
            ).reshape(H_A, W_A, 2, 2)
            precision_fwd = torch.stack((precision_A, precision_AB), dim=-3).reshape(
                -1, 2, 2, 2
            )
            precision_bwd = torch.stack((precision_BA, precision_B), dim=-3).reshape(
                -1, 2, 2, 2
            )
            precision = torch.cat((precision_fwd, precision_bwd), dim=0)
        else:
            precision = None

        grid = get_normalized_grid(1, H_A, W_A, warp.device)[0]
        matches_BA = torch.cat((warp_BA, grid), dim=-1).reshape(-1, 4)
        confidence = torch.cat(
            (confidence_AB.reshape(-1), confidence_BA.reshape(-1)), dim=0
        )
        matches = torch.cat((matches_AB, matches_BA), dim=0)

        expansion_factor = 4
        confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
        if masks is not None:
            confidence = confidence * self._mask_flags(
                masks, H_A, W_A, warp.device
            ).float()
        # multinomial errors when num_samples exceeds the nonzero-probability
        # count (small or empty masks), so clamp and short-circuit on empty.
        num_pool = min(expansion_factor * num_corresp, int((confidence > 0).sum()))
        if num_pool == 0:
            return (
                matches[:0],
                confidence[:0],
                precision[:0][:, 0] if precision is not None else None,
                precision[:0][:, 1] if precision is not None else None,
            )
        corresp_inds = torch.multinomial(confidence, num_pool, replacement=False)
        sampled_matches = matches[corresp_inds]
        sampled_confidence = confidence[corresp_inds]
        if precision is not None:
            sampled_precision = precision[corresp_inds]
        else:
            sampled_precision = None

        density = kde(sampled_matches)

        p = 1 / (density + 1)
        p[density < 10] = (
            1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        )
        balanced_samples = torch.multinomial(
            p, num_samples=min(num_corresp, len(sampled_confidence)), replacement=False
        )
        return (
            sampled_matches[balanced_samples],
            sampled_confidence[balanced_samples],
            sampled_precision[balanced_samples][:, 0]
            if sampled_precision is not None
            else None,
            sampled_precision[balanced_samples][:, 1]
            if sampled_precision is not None
            else None,
        )

    @classmethod
    def to_pixel_coordinates(
        cls, warp: torch.Tensor, H_A: int, W_A: int, H_B: int, W_B: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return to_pixel(warp[..., :2], H=H_A, W=W_A), to_pixel(
            warp[..., 2:], H=H_B, W=W_B
        )
