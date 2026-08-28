# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Low-level TAPIP3D torch.export (pt2) model: encoder and the fused
corr+updater iteration as exported .pt2 programs; the correlation prep
phases and the pointops2 KNN remain in PyTorch (code only). encode_batch runs
one batch of frames (<= seq_len) per call; forward_window runs ONE window of a
full-video input (mirroring ONNXInferenceWrapper._forward_window); image size,
query count, seq_len and the corr pipeline dimensions are derived from the
graph input shapes (derive shapes, do not hardcode).

Dtype-agnostic runtime: the two programs may be exported at fp32 or bf16
(export_pt2.export_encoder/export_iteration with dtype=torch.bfloat16); the
graph dtype is auto-detected from the user placeholder dtypes (curr_coords /
rgb) in __init__ and stored as self.dtype. The eager correlation prep
(pyramids, support features, pointops2 KNN) always runs in fp32; the boundary
casts (rgb in / feats out for the encoder, the iteration program's float
inputs in / delta out) are the only dtype conversions.

CHECKPOINT-FREE CONTRACT: everything learned lives in the two exported
programs (encoder_{H}x{W}.pt2, iteration_{N}.pt2). This module never loads
a checkpoint and never calls models.from_pretrained: the corr processor
used here is the prep-only CorrPrepProcessor (pyramids, support features,
KNN) which has no trainable parameters, and the trained per-iteration math
runs inside the iteration program.

use_uv_input=False contract: the exported iteration graph always includes the
posenc(uv) term (the export-time dim assert forces the full updater-input
layout), so feeding zeros for uv makes the graph apply a constant non-zero
embedding — an approximation matching neither a genuinely no-uv model nor the
uv path. The constructor raises a loud warning; do not treat the flag as
reproducing a no-uv model.

Known visib-logit divergence: in the deterministic regime (fp32, math SDPA)
the exported programs' raw visib logits show a measured systematic max
divergence of ~2.7 (forward) / ~2.0 (bidirectional) vs the eager pipeline —
an established ONNX-wrapper pattern gated at 6.0/4.0 in the equivalence
tests. Do not read the logits as close-tolerance values; the shipped
configuration is unaffected behaviorally."""
from typing import Tuple
import warnings
import torch
from torch import nn
from einops import rearrange, repeat
from torch.export.graph_signature import InputKind

from .utils.corr_prep import CorrPrepProcessor, KNNQueryInput
from .utils._common import batch_unproject
from .utils._norm_utils import (
    compute_normalization_stats, normalize_coords, denormalize_coords,
    build_projector,
)

# ---------------------------------------------------------------------------
# Default model paths (encoder fixed at 480x640 image size; iteration with a
# fixed 1088-query count — 8x8 bbox grid + 32x32 support grid)
# ---------------------------------------------------------------------------
_DEFAULT_ENCODER = "weights/tapip3d/tapip3d_encoder_480x640_bf16.pt2"
_DEFAULT_ITERATION = "weights/tapip3d/tapip3d_iteration_1088_bf16.pt2"

# The CUDA fp32 mem-efficient SDPA kernel (the default fp32 backend on
# Blackwell/sm_120, flash being unavailable for fp32 there) is numerically
# broken for this model's attention tensors (mirror
# export_onnx/export_corr_forward.py). Inside the exported programs SDPA
# decomposes to math ops automatically; pin torch-GPU attention to the math
# backend so eager math matches the graphs.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)


def _user_input_shapes(ep) -> dict:
    """{input name: torch.Size} of the graph's user inputs (no hardcoding)."""
    user_names = {spec.arg.name for spec in ep.graph_signature.input_specs
                  if spec.kind == InputKind.USER_INPUT}
    shapes = {}
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder" and node.name in user_names:
            val = node.meta["val"]
            assert val is not None, f"placeholder {node.name} has no shape metadata"
            shapes[node.name] = val.shape
    assert set(shapes) == user_names, (set(shapes), user_names)
    return shapes


def _user_input_dtypes(ep) -> dict:
    """{input name: torch.dtype} of the graph's user inputs (no hardcoding)."""
    user_names = {spec.arg.name for spec in ep.graph_signature.input_specs
                  if spec.kind == InputKind.USER_INPUT}
    dtypes = {}
    for node in ep.graph_module.graph.nodes:
        if node.op == "placeholder" and node.name in user_names:
            val = node.meta["val"]
            assert val is not None, f"placeholder {node.name} has no dtype metadata"
            dtypes[node.name] = val.dtype
    assert set(dtypes) == user_names, (set(dtypes), user_names)
    return dtypes


class Tapip3D_PT2:
    def __init__(self, encoder_path: str = _DEFAULT_ENCODER,
                 iteration_path: str = _DEFAULT_ITERATION,
                 image_size: Tuple[int, int] = (480, 640),
                 num_iters: int = 6, norm_mode: str = "isotropic",
                 norm_scale: float = 1.0, use_uv_input: bool = True):
        self.num_iters = num_iters
        self.norm_mode = norm_mode
        self.norm_scale = norm_scale
        self.use_uv_input = use_uv_input
        if not use_uv_input:
            warnings.warn(
                "use_uv_input=False feeds zeros for uv, but the exported "
                "iteration graph always includes the posenc(uv) term (the "
                "export-time dim assert forces the full updater-input layout), "
                "so the graph applies a constant non-zero embedding — an "
                "approximation that matches neither a genuinely no-uv model "
                "nor the uv path. Use use_uv_input=True (the shipped "
                "configuration), or re-export an iteration graph from a "
                "genuinely no-uv model if that variant is needed.",
                stacklevel=2,
            )
        # GPU-only (user ruling)
        assert torch.cuda.is_available(), "Tapip3D_PT2 is GPU-only"

        # --- load both programs (checkpoint-free; weights live in the .pt2) ---
        # ExportedModule is stateless-eval by construction (the wrappers were
        # .eval() at export time); it rejects .eval()/ .train() calls.
        ep = torch.export.load(encoder_path)
        self.encoder = ep.module().cuda()
        ep_iter = torch.export.load(iteration_path)
        self.iteration = ep_iter.module().cuda()

        # --- derive all pipeline parameters from the graph input shapes -------
        enc_shapes = _user_input_shapes(ep)
        assert "rgb" in enc_shapes and len(enc_shapes) == 1, enc_shapes.keys()
        enc_T = enc_shapes["rgb"][0]  # static at exactly seq_len (baked in)
        H, W = enc_shapes["rgb"][2], enc_shapes["rgb"][3]  # (16, 3, H, W)
        assert (H, W) == image_size, (
            f"encoder graph is {H}x{W} but image_size arg is {image_size} "
            "(the arg is authoritative; re-export the encoder at this size)")
        self.image_size: Tuple[int, int] = image_size

        it_shapes = _user_input_shapes(ep_iter)
        coords_sh = it_shapes["curr_coords"]  # (1, T, N, 3)
        T, N = coords_sh[1], coords_sh[2]
        assert coords_sh[0] == 1, (
            f"iteration graph is static at batch_size=1 (T={T}, N={N} baked "
            f"in), got batch dim {coords_sh[0]} — re-export the iteration "
            "program at batch size 1")
        assert enc_T == T, (
            f"encoder graph is static at T={enc_T} but the iteration graph at "
            f"T={T} — the two programs must share the same seq_len; re-export "
            "them consistently")
        k = it_shapes["level0_knn_idx"][3]  # (1, T, N, k)
        corr_levels = 0
        while f"level{corr_levels}_feats" in it_shapes:
            corr_levels += 1
        assert corr_levels > 0, "iteration graph has no level0_feats input"
        C = it_shapes["level0_feats"][2]
        for i in range(corr_levels):
            sh = it_shapes[f"level{i}_feats"]  # (1, T, C, h_i, w_i)
            assert sh[1] == T and sh[2] == C, (i, sh)
            h_i, w_i = (H // 4) >> i, (W // 4) >> i
            assert sh[3] == h_i and sh[4] == w_i, (
                f"iteration graph level{i}_feats {sh} does not match the "
                f"(image_size//4)>>{i} = ({h_i},{w_i}) pyramid pattern — "
                "encoder and iteration programs are from different image sizes")
        assert it_shapes["level0_knn_idx"][0] == 1 and \
            it_shapes["level0_knn_idx"][1] == T and \
            it_shapes["level0_knn_idx"][2] == N, it_shapes["level0_knn_idx"]

        self.num_queries = N
        self.seq_len = T

        # --- dtype detection: the two programs are fp32 or bf16, jointly ------
        # derived from the graph placeholder dtypes (fp32 programs keep the
        # original behavior; bf16 programs run the same math in bf16 with the
        # boundary casts in encode_batch/forward_window as the only conversions)
        enc_dtypes = _user_input_dtypes(ep)
        it_dtypes = _user_input_dtypes(ep_iter)
        self.dtype = it_dtypes["curr_coords"]
        assert enc_dtypes["rgb"] == self.dtype, (
            f"encoder graph's rgb input is {enc_dtypes['rgb']} but the "
            f"iteration graph's curr_coords input is {self.dtype} — the two "
            "programs must be exported at the same dtype; re-export them "
            "consistently")

        # --- corr processor: prep phases + pointops2 KNN ----------------------
        # CorrPrepProcessor is the prep-only subset of the original
        # KNNCorrFeature4D_Optimized (pyramids, support features, KNN). The
        # trained per-iteration math runs inside the iteration program, so no
        # weights are ever loaded here.
        self.corr_processor: nn.Module = CorrPrepProcessor(
            image_size=self.image_size,
            corr_levels=corr_levels,
            k_neighbors=k,
        ).cuda().eval()

    # ---------- preprocessing / low-level forwards ---------------------------

    def preprocess(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """(T,3,H,W) in [0,1] -> float32 cuda (T,3,H,W), RAW (no centering).

        The exported encoder graph applies center_rgb (x*2-1) internally
        (ExportEncoderPT2.forward), so pre-centered input would double-center.

        Asserts one batch of frames: T == seq_len (the encoder graph is static
        at exactly seq_len frames), (H,W) == image_size."""
        assert rgb_batch.dim() == 4, rgb_batch.shape
        T, C, H, W = rgb_batch.shape
        assert C == 3, C
        assert T == self.seq_len, (
            f"the encoder graph is static at exactly seq_len={self.seq_len} "
            f"frames (T is baked into the exported program), got T={T}; chunk "
            "longer videos into seq_len-frame encode_batch calls")
        assert (H, W) == self.image_size, ((H, W), self.image_size)
        assert rgb_batch.min() >= -1e-6 and rgb_batch.max() <= 1 + 1e-6, \
            "RGB inputs should be in the range of [0, 1]"
        return rgb_batch.contiguous().float().cuda()

    @torch.inference_mode()
    def encode_batch(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """preprocess + encoder program -> feats (1, T, 128, H/4, W/4) cuda f32.

        The encoder program is static at (seq_len, 3, H, W); T=seq_len (16)
        per call (chunk longer videos, the CNN is per-frame stateless). Only
        output feat_l0 is used; pyramid levels are rebuilt by the corr
        processor's _build_pyramid. The rgb is cast to the graph's dtype
        (self.dtype) at the boundary; the returned feats are cast back to fp32
        so the eager fp32 corr prep downstream is unchanged."""
        out = self.encoder(self.preprocess(rgb_batch).to(self.dtype))
        return out[None].float()

    # ---------- window forward -----------------------------------------------

    @torch.inference_mode()
    def forward_window(self, *, feats, depths, intrs, extrs, queries,
                       window_start: int, window_end: int,
                       coords_init, visibs_init, shared_corr_ctx,
                       depth_roi):
        """Mirror PointTracker3D._wrapped_forward_window for B==1: run the
        fused corr+updater iterations for ONE window of a FULL-VIDEO input.

        feats/depths/intrs/extrs are the full video (B=1, T >= seq_len frames);
        window bounds are absolute (window_start, window_end) with
        window_end - window_start == seq_len. The normalization stats use the
        window frames only, the correlation context is prepared over the full
        video and time-sliced to the window, and the shared support context
        (prepared by the caller over the same full-video tensors) supplies
        support features sampled at the queries' true home frames — exactly
        like ONNXInferenceWrapper._forward_window. The corr prep and the
        coords/visibs accumulators run in fp32; the iteration program's float
        inputs are cast to its dtype (self.dtype, fp32 or bf16) at the
        boundary and the returned delta is cast back to fp32.
        Returns denormalized (coords (1, W, N, 3) world, visibs (1, W, N) logits)."""
        assert window_end - window_start == self.seq_len, \
            (window_start, window_end, self.seq_len)

        inv_extrinsics = torch.linalg.inv(extrs)
        camera_locs = inv_extrinsics[..., :3, 3]  # (B, T, 3)

        pcds = batch_unproject(depths, intrs, extrs)

        # window normalization stats (window frames only, mirroring
        # PointTracker3D._wrapped_forward_window lines 355-380)
        _pcds = pcds[:, window_start:window_end].clone()
        _pcds[(depths[:, window_start:window_end] == 0)[:, :, None]
              .expand(-1, -1, 3, -1, -1)] = torch.nan
        if depth_roi is not None:
            depth_roi = depth_roi.reshape(2)
            _pcds.masked_fill_(
                depths[:, window_start:window_end, None] > depth_roi[1], torch.nan)
            _pcds.masked_fill_(
                depths[:, window_start:window_end, None] < depth_roi[0], torch.nan)
        mean_coords, std_coords = compute_normalization_stats(_pcds, norm_mode=self.norm_mode)

        normalized_queries = queries.clone()
        normalized_queries[..., 1:] = normalize_coords(
            queries[..., 1:], mean_coords, std_coords, self.norm_scale)
        normalized_coords_init = normalize_coords(
            coords_init, mean_coords, std_coords, self.norm_scale)
        normalized_camera_locs = (camera_locs - mean_coords[:, None, :]) / std_coords[:, None, :]
        normalized_pcds = (pcds - mean_coords[:, :, None, None]) / std_coords[:, :, None, None] * self.norm_scale

        projector = build_projector(intrs, extrs, mean_coords, std_coords,
                                    self.norm_scale, self.image_size,
                                    norm_mode=self.norm_mode)
        corr_ctx = self.corr_processor.prepare_window(
            feats=feats, camera_locs=normalized_camera_locs, pcds=normalized_pcds,
            queries=normalized_queries, projector=projector, shared_ctx=shared_corr_ctx,
        )
        corr_ctx = corr_ctx.time_slice(start=window_start, end=window_end)

        coords = normalized_coords_init.clone()
        visibs = visibs_init.clone()
        window_projector = lambda x: projector(
            torch.cat([x[..., :1] + window_start, x[..., 1:]], dim=-1))
        corr = self.corr_processor

        for _ in range(self.num_iters):
            coords = coords.detach().clone()
            visibs = visibs.detach().clone()
            B, T, N, _ = coords.shape

            # 1. eager pointops2 KNN (mirror the ONNX wrapper's
            #    _run_corr_forward KNN block; the trained math runs inside
            #    the iteration program)
            knn_inputs = []
            for i in range(corr.corr_levels):
                map_pcds = rearrange(corr_ctx.pcd_pyramid[i], "b t c h w -> (b t) (h w) c")
                curr_coords_ = rearrange(coords, "b t n c -> (b t) n c")
                knn_inputs.append(KNNQueryInput(
                    query_coords=curr_coords_.reshape(-1, 3),
                    context_coords=map_pcds.reshape(-1, 3),
                    query_batch_offsets=(torch.arange(curr_coords_.shape[0],
                                        device=curr_coords_.device, dtype=torch.int32) + 1)
                                        * curr_coords_.shape[1],
                    context_batch_offsets=(torch.arange(map_pcds.shape[0],
                                          device=map_pcds.device, dtype=torch.int32) + 1)
                                          * map_pcds.shape[1],
                ))
            knn_outputs = corr.multi_knnquery(corr.k_neighbors, knn_inputs)

            # 2. iteration program inputs: level tensors + coords/visibs/uv/mask.
            #    The program's float inputs are cast to its dtype (self.dtype)
            #    at the boundary — the fp32 accumulators below are never mutated,
            #    only the copies handed to the graph are cast. knn_idx (int64)
            #    and mask (bool) pass through unchanged.
            inputs = {}
            for i in range(corr.corr_levels):
                hw = corr_ctx.pcd_pyramid[i].shape[-2] * corr_ctx.pcd_pyramid[i].shape[-1]
                knn_idx = (knn_outputs[i] % hw).reshape(B, T, N, corr.k_neighbors).long()
                inputs[f"level{i}_feats"] = corr_ctx.feat_pyramid[i].to(self.dtype)
                inputs[f"level{i}_pcds"] = corr_ctx.pcd_pyramid[i].to(self.dtype)
                inputs[f"level{i}_knn_idx"] = knn_idx
                inputs[f"level{i}_support_feats"] = corr_ctx.query_support_feat_pyramid[i].to(self.dtype)
                inputs[f"level{i}_support_offsets"] = corr_ctx.query_support_offset_pyramid[i].to(self.dtype)
            inputs["curr_coords"] = coords.to(self.dtype)
            inputs["visibs"] = visibs.to(self.dtype)
            # uv: normalized pixel coords in [0, 1] via the window projector on
            # coords_with_time, exactly as assemble_updater_input computes them
            # (repeat(arange(T)) + coords, project, reshape, [..., :2],
            # divide by (W-1, H-1)); zeros when use_uv_input=False.
            if self.use_uv_input:
                coords_with_time = torch.cat([
                    repeat(torch.arange(T, device=coords.device, dtype=coords.dtype),
                           "t -> b t n 1", b=B, n=N),
                    coords,
                ], dim=-1)
                pixel_coords = rearrange(
                    window_projector(rearrange(coords_with_time, "b t n c -> b (t n) c")),
                    "b (t n) c -> b t n c", t=T)[..., :2]
                uv = pixel_coords / torch.tensor(
                    [self.image_size[1] - 1, self.image_size[0] - 1],
                    device=pixel_coords.device)
            else:
                uv = torch.zeros(B, T, N, 2, device=coords.device)
            inputs["uv"] = uv.to(self.dtype)
            inputs["mask"] = torch.ones(T, N, dtype=torch.bool, device=coords.device)

            # 3. one fused corr+updater iteration; the returned delta is cast
            #    back to fp32 so the coords/visibs accumulators stay fp32 across
            #    iterations (their fp32 values feed the next iteration's KNN and
            #    boundary casts)
            delta = self.iteration(**inputs).float()
            coords = coords + delta[..., :3]
            visibs = visibs + delta[..., 3]

        return self.postprocess(coords, visibs, mean_coords, std_coords)

    def postprocess(self, coords, visibs, mean_coords, std_coords):
        """Output transform after the iteration loop: denormalize coords to
        world; visibs pass through unchanged (logits)."""
        return denormalize_coords(coords, mean_coords, std_coords, self.norm_scale), visibs
