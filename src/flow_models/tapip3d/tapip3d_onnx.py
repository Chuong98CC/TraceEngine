# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Low-level TAPIP3D ONNX model: encoder, point updater and the corr
pipeline's trained per-iteration forward as ORT sessions; the correlation
prep phases and the pointops2 KNN remain in PyTorch (code only, no
checkpoint). Runs forward for ONE batch of frames (<= seq_len) per call;
image size, query count, seq_len and the corr pipeline dimensions are
auto-derived from the ONNX graphs (derive_model_params)."""

from typing import Tuple
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from einops import rearrange

from .utils.corr_prep import CorrPrepProcessor, KNNQueryInput
from .utils.common_utils import batch_unproject
from .utils.correlation_utils import (
    compute_normalization_stats, normalize_coords, denormalize_coords,
    build_projector, interpolate_time_embed, assemble_updater_input,
    derive_model_params,
)

class Tapip3D_ONNX:
    def __init__(self, encoder_path, updater_path, corr_forward_path,
                 num_iters: int = 6, norm_mode: str = "isotropic",
                 norm_scale: float = 1.0):
        self.num_iters = num_iters
        self.norm_mode = norm_mode
        self.norm_scale = norm_scale

        assert torch.cuda.is_available(), "Tapip3D_ONNX is GPU-only"
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
        self.encoder_sess = ort.InferenceSession(encoder_path, providers=providers)
        self.updater_sess = ort.InferenceSession(updater_path, providers=providers)

        self._corr_forward_path = corr_forward_path
        self._providers = providers
        # construct once here for the shape-consistency asserts, then release
        self.corr_forward_sess = ort.InferenceSession(corr_forward_path,
                                                      providers=providers)

        # --- derive all pipeline parameters from the ONNX graphs ---------------
        params = derive_model_params(
            rgb_shape=self.encoder_sess.get_inputs()[0].shape,
            upd_shape=self.updater_sess.get_inputs()[0].shape,
            graph_inputs={inp.name: inp.shape
                          for inp in self.corr_forward_sess.get_inputs()},
            graph_outputs=[out.shape
                           for out in self.corr_forward_sess.get_outputs()],
        )
        self.image_size: Tuple[int, int] = params.image_size
        self.num_queries = params.num_queries
        self.seq_len = params.seq_len
        self.use_uv_input = params.use_uv_input
        self.time_emb = params.time_emb.cuda()

        # --- corr processor: prep phases + pointops2 KNN ----------------------
        # CorrPrepProcessor is the prep-only subset of the original
        # KNNCorrFeature4D_Optimized (pyramids, support features, KNN). The
        # trained per-iteration math runs inside the corr_forward ONNX graph,
        # so no weights are ever loaded here.
        self.corr_processor: nn.Module = CorrPrepProcessor(
            image_size=self.image_size,
            corr_levels=params.corr_levels,
            k_neighbors=params.k_neighbors,
        ).cuda().eval()

        del self.corr_forward_sess
        self.corr_forward_sess = None
        torch.cuda.empty_cache()

    # ---------- preprocessing / low-level forwards ---------------------------

    def preprocess(self, rgb_batch: torch.Tensor) -> np.ndarray:
        """(T,3,H,W) in [0,1] -> float32 numpy (T,3,H,W), RAW (no centering).

        The exported encoder graph applies center_rgb (x*2-1) internally
        (ExportEncoder.forward), so pre-centered input would double-center.
        The reference wrapper's _encode feeds raw frames for the same reason.

        Asserts one batch of frames: T <= seq_len, (H,W) == image_size."""
        assert rgb_batch.dim() == 4, rgb_batch.shape
        T, C, H, W = rgb_batch.shape
        assert C == 3, C
        assert T <= self.seq_len, \
            f"one batch is at most seq_len={self.seq_len} frames, got {T}"
        assert (H, W) == self.image_size, ((H, W), self.image_size)
        assert rgb_batch.min() >= -1e-6 and rgb_batch.max() <= 1 + 1e-6, \
            "RGB inputs should be in the range of [0, 1]"
        return rgb_batch.contiguous().float().cpu().numpy()

    def encode_batch(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """preprocess + encoder session -> feats (1, T, 128, H/4, W/4) cuda f32.

        Only output feat_l0 is used; pyramid levels are rebuilt by the corr
        processor's _build_pyramid."""
        out = self.encoder_sess.run(None, {"rgb": self.preprocess(rgb_batch)})[0]
        return torch.from_numpy(out).cuda().float()[None]

    # ---------- window forward -----------------------------------------------

    @torch.inference_mode()
    def forward_window(self, *, feats, depths, intrs, extrs, queries,
                       coords_init, visibs_init, shared_corr_ctx,
                       depth_roi):
        """ONE batch (16 frames). feats/depths/intrs/extrs are the 17-frame
        stack [frame0] + [16 window frames] with B=1; window bounds are the
        local (1, 17). Returns denormalized (coords (1,16,N,3) world,
        visibs (1,16,N) logits)."""
        B, T_stack = feats.shape[:2]
        assert T_stack == self.seq_len + 1, (T_stack, self.seq_len + 1)
        W = self.seq_len
        window_start, window_end = 1, self.seq_len + 1

        # lazy per-window corr_forward session (see __init__ comment)
        if self.corr_forward_sess is None:
            self.corr_forward_sess = ort.InferenceSession(
                self._corr_forward_path, providers=self._providers)

        inv_extrinsics = torch.linalg.inv(extrs)
        camera_locs = inv_extrinsics[..., :3, 3]  # (B, 17, 3)

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

        for _ in range(self.num_iters):
            coords = coords.detach().clone()
            visibs = visibs.detach().clone()
            corr_embs = self._run_corr_forward(corr_ctx, coords)
            updater_input = assemble_updater_input(
                visibs=visibs, corr_embs=corr_embs, coords=coords,
                projector=window_projector, image_size=self.image_size,
                use_uv_input=self.use_uv_input,
            )
            updater_input = updater_input + interpolate_time_embed(self.time_emb, W)[:, :, None, :]
            delta = self._run_updater(updater_input)
            coords = coords + delta[..., :3]
            visibs = visibs + delta[..., 3]

        out = self.postprocess(coords, visibs, mean_coords, std_coords)
        del self.corr_forward_sess
        self.corr_forward_sess = None
        torch.cuda.empty_cache()
        return out

    def postprocess(self, coords, visibs, mean_coords, std_coords):
        """Output transform after the iteration loop: denormalize coords to
        world; visibs pass through unchanged (logits)."""
        return denormalize_coords(coords, mean_coords, std_coords, self.norm_scale), visibs

    def _run_updater(self, updater_input: torch.Tensor) -> torch.Tensor:
        """updater_input (1, 16, N, 777) -> delta (1, 16, N, 5) at the graph's
        static shape. The exported graph is fully static — feed the full 4-D
        tensor, do NOT strip the batch dim (ORT would reject it)."""
        mask = np.ones((self.seq_len, self.num_queries), dtype=bool)
        out = self.updater_sess.run(
            None, {"updater_input": updater_input.contiguous().cpu().numpy(),
                   "mask": mask})[0]
        return torch.from_numpy(out).cuda()

    def _run_corr_forward(self, corr_ctx, coords) -> torch.Tensor:
        """Per-iteration correlation forward: the pointops2 KNN runs in torch
        (exactly KNNCorrFeature4D_Optimized.forward()'s KNN block), then the
        trained per-level math runs as the corr_forward ONNX graph.

        coords: (1, seq_len, N, 3) normalized. corr_ctx: the window-sliced
        context (pyramids are (1, seq_len, ...)). Returns corr_embs
        (1, seq_len, N, corr_levels * transformer_dim)."""
        corr = self.corr_processor
        B, T, N, _ = coords.shape

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

        feeds = {}
        for i in range(corr.corr_levels):
            hw = corr_ctx.pcd_pyramid[i].shape[-2] * corr_ctx.pcd_pyramid[i].shape[-1]
            knn_idx = (knn_outputs[i] % hw).reshape(B, T, N, corr.k_neighbors).long()
            feeds[f"level{i}_feats"] = corr_ctx.feat_pyramid[i].contiguous().cpu().numpy()
            feeds[f"level{i}_pcds"] = corr_ctx.pcd_pyramid[i].contiguous().cpu().numpy()
            feeds[f"level{i}_knn_idx"] = knn_idx.contiguous().cpu().numpy()
            feeds[f"level{i}_support_feats"] = \
                corr_ctx.query_support_feat_pyramid[i].contiguous().cpu().numpy()
            feeds[f"level{i}_support_offsets"] = \
                corr_ctx.query_support_offset_pyramid[i].contiguous().cpu().numpy()
        feeds["curr_coords"] = coords.contiguous().cpu().numpy()
        out = self.corr_forward_sess.run(None, feeds)[0]
        return torch.from_numpy(out).cuda()
