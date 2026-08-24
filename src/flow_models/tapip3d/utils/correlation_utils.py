# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Python-side math helpers for the ONNX inference wrapper.

These mirror the non-exportable parts of PointTracker3D / KNNCorrFeature4D_Optimized
exactly, so ONNX results are numerically equivalent to the PyTorch pipeline.
"""
from dataclasses import dataclass
from typing import Callable, Tuple
import torch
import torch.nn.functional as F

from .cotracker_utils import posenc, get_1d_sincos_pos_embed_from_grid  # noqa: F401 (re-export)

EPS = 1e-6

def nanvar(tensor, dim=None, keepdim=False):
    # Verbatim copy of the shim in models/point_tracker_3d.py (torch.nanstd does
    # not exist; this is the exact math the pipeline uses for normalization).
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output

def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output

def compute_normalization_stats(pcds: torch.Tensor, norm_mode: str = "isotropic") -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror PointTracker3D._wrapped_forward_window (lines 355-380).

    pcds: (B, T_w, 3, H, W) with invalid entries already set to NaN.
    Returns mean_coords (B, 3), std_coords (B, 3).
    """
    assert norm_mode in ["anisotropy", "isotropic"], norm_mode
    if norm_mode == "anisotropy":
        mean_coords = torch.nanmean(pcds, dim=(1, 3, 4), keepdim=True)  # (B, 1, 3, 1, 1)
        std_coords = nanstd(pcds, dim=(1, 3, 4), keepdim=True)
    else:
        mean_coords = torch.nanmean(pcds, dim=(1, 3, 4), keepdim=True)
        std_coords = nanstd(pcds - mean_coords, dim=(1, 2, 3, 4), keepdim=True).expand(-1, -1, 3, -1, -1)
    return mean_coords.reshape(-1, 3), std_coords.reshape(-1, 3)

def normalize_coords(coords: torch.Tensor, mean_coords: torch.Tensor, std_coords: torch.Tensor, norm_scale: float) -> torch.Tensor:
    return (coords - mean_coords[:, None, :]) / std_coords[:, None, :] * norm_scale

def denormalize_coords(coords: torch.Tensor, mean_coords: torch.Tensor, std_coords: torch.Tensor, norm_scale: float) -> torch.Tensor:
    return (coords / norm_scale * std_coords[:, None, :]) + mean_coords[:, None, :]

def build_projector(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    mean_coords: torch.Tensor,
    std_coords: torch.Tensor,
    norm_scale: float,
    image_size: Tuple[int, int],
    norm_mode: str = "isotropic",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a projector mirroring PointTracker3D._project.

    Input points: (B, N, 4) = (t, x, y, z) in normalized space.
    Output: (B, N, 3) = (clamped_x, clamped_y, camera_z).
    """
    def projector(points: torch.Tensor) -> torch.Tensor:
        B, N, _ = points.shape
        time_index = points[..., 0].long()
        coords = points[..., 1:]
        batch_index = torch.arange(B, device=points.device)[:, None].expand(-1, N)
        K = intrinsics[batch_index, time_index]   # B, N, 3, 3
        E = extrinsics[batch_index, time_index]   # B, N, 4, 4

        if norm_mode in ["anisotropy", "isotropic"]:
            coords = (coords / norm_scale * std_coords[:, None, :]) + mean_coords[:, None, :]

        coords_homo = torch.cat([coords, torch.ones_like(coords[..., :1])], dim=-1)
        coords_local_homo = torch.einsum("bnij,bnj->bni", E, coords_homo)
        coords_local = coords_local_homo[..., :3] / torch.clamp(coords_local_homo[..., 3:], min=EPS)
        coords_pixel = torch.einsum("bnij,bnj->bni", K, coords_local)
        coords_pixel = coords_pixel[..., :2] / torch.clamp(coords_pixel[..., 2:3], min=EPS)

        clamped_x = torch.clamp(coords_pixel[..., 0], min=-image_size[1] * 2, max=image_size[1] * 2)
        clamped_y = torch.clamp(coords_pixel[..., 1], min=-image_size[0] * 2, max=image_size[0] * 2)
        return torch.cat([clamped_x[..., None], clamped_y[..., None], coords_local[..., 2:]], dim=-1)
    return projector

def interpolate_time_embed(time_emb: torch.Tensor, T: int) -> torch.Tensor:
    previous_dtype = time_emb.dtype
    T_emb = time_emb.shape[1]
    if T == T_emb:
        return time_emb
    time_emb_f = time_emb.float()
    time_emb_f = F.interpolate(time_emb_f.permute(0, 2, 1), size=T, mode="linear").permute(0, 2, 1)
    return time_emb_f.to(previous_dtype)

def assemble_updater_input(
    *,
    visibs: torch.Tensor,      # (B, T, N)
    corr_embs: torch.Tensor,   # (B, T, N, 512)
    coords: torch.Tensor,      # (B, T, N, 3) normalized
    projector: Callable[[torch.Tensor], torch.Tensor],
    image_size: Tuple[int, int],
    use_uv_input: bool,
) -> torch.Tensor:
    """Mirror PointTracker3D._forward_window_iter lines 191-222 (before time embedding).
    use_local_pos_input is not supported."""
    from einops import rearrange, repeat
    B, T, N = coords.shape[:3]
    updater_input = [visibs[..., None], corr_embs]

    rel_coords_forward = coords[:, :-1] - coords[:, 1:]
    rel_coords_backward = coords[:, 1:] - coords[:, :-1]
    rel_coords_forward = F.pad(rel_coords_forward, (0, 0, 0, 0, 0, 1))
    rel_coords_backward = F.pad(rel_coords_backward, (0, 0, 0, 0, 1, 0))
    rel_pos_emb_input = posenc(torch.cat([rel_coords_forward, rel_coords_backward], dim=-1), min_deg=-2, max_deg=14)
    updater_input.append(rel_pos_emb_input)

    if use_uv_input:
        coords_with_time = torch.cat([
            repeat(torch.arange(T, device=coords.device, dtype=coords.dtype), "t -> b t n 1", b=B, n=N),
            coords,
        ], dim=-1)
        pixel_coords = rearrange(projector(rearrange(coords_with_time, "b t n c -> b (t n) c")), "b (t n) c -> b t n c", t=T)[..., :2]
        normalized_pixel_coords = pixel_coords / torch.tensor([image_size[1] - 1, image_size[0] - 1], device=pixel_coords.device)
        pixel_pos_emb_input = posenc(normalized_pixel_coords, min_deg=-2, max_deg=14)
        updater_input.append(pixel_pos_emb_input)

    return torch.cat(updater_input, dim=-1)

@dataclass
class ModelParams:
    """Pipeline parameters derived from the three ONNX graphs."""
    image_size: Tuple[int, int]
    seq_len: int
    num_queries: int
    feat_dim: int
    k_neighbors: int
    corr_levels: int
    transformer_dim: int
    use_local_pos_input: bool
    use_uv_input: bool
    updater_input_dim: int
    time_emb: torch.Tensor  # (1, seq_len, updater_input_dim) float32, CPU


def derive_model_params(*, rgb_shape, upd_shape, graph_inputs, graph_outputs) -> ModelParams:
    """Derive every pipeline parameter the ONNX graphs encode, and verify the
    three graphs are mutually consistent (mixed-artifact pairing fails here).

    rgb_shape: encoder input shape from sess.get_inputs()[0].shape
    upd_shape: updater input shape from sess.get_inputs()[0].shape
    graph_inputs: {name: shape} of the corr_forward graph inputs
    graph_outputs: [shape] of the corr_forward graph outputs
    Pure function: no ORT, no GPU, no checkpoint.
    """
    # --- encoder: image size --------------------------------------------------
    H, W = rgb_shape[2], rgb_shape[3]
    assert isinstance(H, int) and isinstance(W, int), \
        f"encoder ONNX must have static H/W, got {rgb_shape}"

    # --- updater: seq_len / num_queries / input dim ---------------------------
    upd_B, upd_T, upd_N = upd_shape[0], upd_shape[1], upd_shape[2]
    assert upd_B == 1, (
        f"updater ONNX must be exported at batch 1, got batch={upd_B}. "
        "batch > 1 is for external runtimes.")
    updater_input_dim = upd_shape[3]

    def _g(name):
        assert name in graph_inputs, f"corr_forward graph missing input {name}"
        return graph_inputs[name]

    l0_feats = _g("level0_feats")            # (1, seq_len, feat_dim, h0, w0)
    l0_knn = _g("level0_knn_idx")            # (1, seq_len, N, k)
    l0_sup = _g("level0_support_feats")      # (1, N, k+1, feat_dim)
    l0_offs = _g("level0_support_offsets")   # (1, N, k+1, 3) or (..., 4)
    coords_sh = _g("curr_coords")            # (1, seq_len, N, 3)
    assert len(graph_outputs) == 1, f"expected one corr_forward output, got {graph_outputs}"
    out_sh = graph_outputs[0]

    feat_dim = l0_feats[2]
    k_neighbors = l0_knn[3]
    corr_levels = 0
    while f"level{corr_levels}_feats" in graph_inputs:
        corr_levels += 1
    assert corr_levels > 0, "corr_forward graph has no level0_feats input"
    assert out_sh[3] % corr_levels == 0, \
        f"corr_forward output dim {out_sh[3]} is not divisible by corr_levels={corr_levels}"
    transformer_dim = out_sh[3] // corr_levels

    # --- cross-graph consistency (mismatched artifacts fail here) -------------
    assert l0_feats[1] == upd_T, (
        f"corr_forward ONNX has seq_len={l0_feats[1]} but updater ONNX has "
        f"seq_len={upd_T} — mismatched artifacts")
    assert l0_knn[1] == upd_T and l0_knn[2] == upd_N, (
        f"corr_forward ONNX knn_idx is {l0_knn[1:3]} but updater ONNX has "
        f"(seq_len={upd_T}, num_queries={upd_N}) — mismatched artifacts")
    assert l0_sup[1] == upd_N and l0_sup[2] == k_neighbors + 1 and l0_sup[3] == feat_dim, (
        f"corr_forward ONNX support_feats {l0_sup} inconsistent with "
        f"num_queries={upd_N}, k_neighbors={k_neighbors}, feat_dim={feat_dim}")
    assert coords_sh[1] == upd_T and coords_sh[2] == upd_N, (
        f"corr_forward ONNX curr_coords {coords_sh} inconsistent with updater "
        f"(seq_len={upd_T}, num_queries={upd_N})")
    assert out_sh[1] == upd_T and out_sh[2] == upd_N, (
        f"corr_forward ONNX output {out_sh} inconsistent with updater "
        f"(seq_len={upd_T}, num_queries={upd_N})")
    for i in range(corr_levels):
        pcds_sh = _g(f"level{i}_pcds")       # (1, seq_len, 3, h_i, w_i)
        h_i, w_i = (H // 4) >> i, (W // 4) >> i
        assert pcds_sh[3] == h_i and pcds_sh[4] == w_i, (
            f"corr_forward ONNX level{i}_pcds {pcds_sh} does not match the "
            f"(image_size//4)>>{i} = ({h_i},{w_i}) pyramid pattern")

    # --- flags the graphs encode ----------------------------------------------
    supp_offs_dim = l0_offs[3]
    assert supp_offs_dim in (3, 4), (
        f"corr_forward ONNX level0_support_offsets last dim must be 3 "
        f"(use_local_pos_input=False) or 4 (True), got {supp_offs_dim}")
    use_local_pos_input = supp_offs_dim == 4
    assert not use_local_pos_input, "use_local_pos_input is not supported"

    def posenc_dim(n: int) -> int:
        # actual output width of the runtime posenc (no hardcoded constants)
        return posenc(torch.zeros(1, n), min_deg=-2, max_deg=14).shape[-1]

    corr_out_dim = corr_levels * transformer_dim
    candidate_no_uv = 1 + corr_out_dim + posenc_dim(6)
    candidate_uv = candidate_no_uv + posenc_dim(2)
    assert updater_input_dim in (candidate_no_uv, candidate_uv), (
        f"updater ONNX input dim {updater_input_dim} matches neither assembled "
        f"updater input candidate (no uv: {candidate_no_uv}, with uv: "
        f"{candidate_uv}) — mismatched artifacts or an unsupported input layout")
    use_uv_input = updater_input_dim == candidate_uv

    # --- deterministic time embedding (same math as PointTracker3D.__init__) --
    time_grid = torch.linspace(0, upd_T - 1, upd_T)
    time_emb = get_1d_sincos_pos_embed_from_grid(updater_input_dim, time_grid)

    return ModelParams(
        image_size=(H, W), seq_len=upd_T, num_queries=upd_N,
        feat_dim=feat_dim, k_neighbors=k_neighbors, corr_levels=corr_levels,
        transformer_dim=transformer_dim,
        use_local_pos_input=use_local_pos_input, use_uv_input=use_uv_input,
        updater_input_dim=updater_input_dim, time_emb=time_emb,
    )
