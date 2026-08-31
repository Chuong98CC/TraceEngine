"""Shared helper functions for the RoMaV2 .pt2 wrapper and matching tools.

Pure tensor/geometry helpers used by both ``romav2.RoMaV2PT2`` (grid
construction, sampling, KDE, coordinate conversion) and the multi-image
matching tools (dense-warp sampling, overlap sampling, cycle-error
filtering).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_normalized_grid(
    B: int, H: int, W: int, device: torch.device
) -> torch.Tensor:
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    return torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)


def bhwc_grid_sample(
    x: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    return F.grid_sample(
        x.permute(0, 3, 1, 2), grid, mode=mode, align_corners=align_corners
    ).permute(0, 2, 3, 1)


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    # gaussian kernel density estimate
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    return scores.sum(dim=-1)


def to_pixel(x: torch.Tensor, *, H: int, W: int) -> torch.Tensor:
    return torch.stack(((x[..., 0] + 1) / 2 * W, (x[..., 1] + 1) / 2 * H), dim=-1)


def warp_points(warp: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Sample a dense warp field at query points.

    warp: (1, H, W, 2) normalized destination coordinates (x, y), as returned
        by RoMaV2PT2.match (warp_AB).
    pts: (M, 2) normalized source coordinates in [-1, 1].
    Returns (M, 2) normalized coordinates in the other image.
    """
    out = F.grid_sample(
        warp.permute(0, 3, 1, 2),  # (1, 2, H, W)
        pts[None, None],  # (1, M, 1, 2)
        mode="bilinear",
        align_corners=False,
    )  # (1, 2, 1, M)
    return out[0, :, 0, :].mT  # (M, 2)


def sample_overlap(overlap: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Sample an overlap map at query points.

    overlap: (1, H, W) or (1, H, W, 1) confidence map, as returned by
        RoMaV2PT2.match (overlap_AB).
    pts: (M, 2) normalized coordinates in [-1, 1].
    Returns (M,) overlap values.
    """
    if overlap.ndim == 4:
        overlap = overlap[..., 0]  # (1, H, W)
    out = F.grid_sample(
        overlap[None],  # (1, 1, H, W)
        pts[None, None],  # (1, M, 1, 2)
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, 1, M)
    return out[0, 0, 0, :]  # (M,)


def compute_cycle_errors(
    positions: torch.Tensor,
    pair_warps: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Per-point max round-trip inconsistency over all image pairs.

    positions: (M, N, 2) normalized tracked positions, one column per image.
    pair_warps: {(i, j): (warp_ij, warp_ji)} for every pair i < j, warps of
        shape (1, H, W, 2).
    Returns (M,) -- for each point, the max over pairs of
    max(|warp_ij(p_i) - p_j|, |warp_ji(p_j) - p_i|) in normalized coords.
    """
    errs = []
    for (i, j), (warp_ij, warp_ji) in pair_warps.items():
        fwd = (warp_points(warp_ij, positions[:, i]) - positions[:, j]).norm(
            dim=-1
        )
        bwd = (warp_points(warp_ji, positions[:, j]) - positions[:, i]).norm(
            dim=-1
        )
        errs.append(torch.maximum(fwd, bwd))
    return torch.stack(errs).amax(dim=0)  # (M,)


def filter_matches(
    overlaps: torch.Tensor,
    roundtrip_err: torch.Tensor | None = None,
    overlap_th: float = 0.5,
    cycle_th: float = 0.01,
) -> torch.Tensor:
    """Mask of points visible in every image.

    overlaps: (M, N) per-point overlap confidence in each image.
    roundtrip_err: (M,) per-point cycle error, or None to skip the check.
    Returns (M,) bool -- points with overlap above overlap_th in every image
    and (if given) round-trip error below cycle_th.
    """
    mask = (overlaps > overlap_th).all(dim=-1)
    if roundtrip_err is not None:
        mask = mask & (roundtrip_err < cycle_th)
    return mask
