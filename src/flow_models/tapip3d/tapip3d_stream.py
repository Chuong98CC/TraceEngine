# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Streaming inference over a Tapip3D_ONNX model.

Mirrors stream_inference.StreamInference.run() exactly — same window
scheduling (plan_windows), same carry/frame0 state machine — while the
low-level forwards (one batch of frames per encoder call, one window per
forward_window call) run on the ONNX model. Unlike StreamInference there is
NO full-video buffer: that allocation exists only for cudnn-bf16
view-sensitivity of the PyTorch encoder, which the fp32 ORT encoder does not
have.
"""

from typing import Optional, Tuple
import torch
from einops import repeat
from dataclasses import dataclass
from .utils.common_utils import batch_unproject

def plan_windows(total_frames: int, seq_len: int) -> tuple[list[tuple[int, int]], int]:
    """Return (windows, pad): every window (start, end) in execution order.

    Mirrors streaming_forward's padding: pad T to a multiple of seq_len//2.
    """
    stride = seq_len // 2
    pad = (stride - total_frames % stride) % stride
    T = total_frames + pad
    windows = [(we - seq_len, we) for we in range(seq_len, T + 1, stride)]
    return windows, pad

@dataclass
class Prediction:
    coords: torch.Tensor # (B, T, N, 3)
    visibs: torch.Tensor # (B, T, N)
    confs: Optional[torch.Tensor] = None # (B, T, N)

    def __post_init__(self):
        assert not self.coords.requires_grad and not self.visibs.requires_grad

    def to(self, device: str):
        return Prediction(
            coords=self.coords.to(device),
            visibs=self.visibs.to(device),
            confs=self.confs.to(device) if self.confs is not None else None,
        )

    def time_slice(self, start: int, end: int):
        assert start >= 0 and end <= self.coords.shape[1] and start < end, "the range of start and end is out of bounds"
        return Prediction(
            coords=self.coords[:, start:end],
            visibs=self.visibs[:, start:end],
            confs=self.confs[:, start:end] if self.confs is not None else None,
        )

    def query_slice(self, s: slice):
        return Prediction(
            coords=self.coords[:, :, s],
            visibs=self.visibs[:, :, s],
            confs=self.confs[:, :, s] if self.confs is not None else None,
        )

class Tapip3DStreamONNX:
    def __init__(self, onnx_model, queries: torch.Tensor,
                 depth_roi: Optional[torch.Tensor] = None,
                 device: str = "cuda"):
        self.onnx_model = onnx_model
        self.queries = queries.to(device)
        assert self.queries.shape[0] == onnx_model.num_queries, (
            f"exact-N contract: queries ({self.queries.shape[0]}) must match "
            f"the ONNX model's num_queries ({onnx_model.num_queries})")
        assert (self.queries[..., 0] < self.onnx_model.seq_len).all(), (
            "StreamONNX requires all query frames < seq_len "
            f"({self.onnx_model.seq_len}) — the static updater graph cannot "
            "mask late-frame queries")
        self.depth_roi = None if depth_roi is None else depth_roi.to(device)
        self.device = device
        self.seq_len = onnx_model.seq_len
        self.stride = onnx_model.seq_len // 2

    @torch.inference_mode()
    def run(self, batches, total_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all windows; `batches` yields CPU tuples
        (video (T,3,H,W) in [0,1], depths, intrs, extrs). Returns
        (coords (total_frames, N, 3), visibs (total_frames, N) logits).
        All query frames must be < seq_len (asserted in __init__);
        late-frame queries would be window-masked, which the static updater
        graph cannot represent."""
        windows, pad = plan_windows(total_frames, self.seq_len)
        T = total_frames + pad

        B, N = 1, self.queries.shape[0]
        query_point = self.queries[None]
        query_coords = query_point[..., 1:]
        query_frames = query_point[..., 0].long()

        pred = Prediction(
            coords=repeat(query_coords, "b n c -> b t n c", t=T).clone(),
            visibs=torch.zeros(B, T, N, device=self.device, dtype=query_point.dtype),
        )

        frame0 = None            # (feats, depths, intrs, extrs) for global frame 0
        shared_corr_ctx = None
        carry = None             # (feats, depths, intrs, extrs) for last stride frames
        n_windows = 0
        k = 0
        for video_b, depths_b, intrs_b, extrs_b in batches:
            depths_b = depths_b.to(self.device)
            intrs_b = intrs_b.to(self.device)
            extrs_b = extrs_b.to(self.device)

            # pad a short final batch up to the padded length T (a multiple of
            # seq_len // 2), mirroring StreamInference.run()
            n = video_b.shape[0]
            pad_n = 0
            if n < self.seq_len:
                pad_n = T - self.seq_len * k - n
                assert pad_n >= 0, f"batch {k} extends past padded length {T}"
                if pad_n > 0:
                    video_b = torch.cat([video_b, video_b[-1:].expand(pad_n, -1, -1, -1)], 0)
                    depths_b = torch.cat([depths_b, depths_b[-1:].expand(pad_n, -1, -1)], 0)
                    intrs_b = torch.cat([intrs_b, intrs_b[-1:].expand(pad_n, -1, -1)], 0)
                    extrs_b = torch.cat([extrs_b, extrs_b[-1:].expand(pad_n, -1, -1)], 0)

            # ONE batch through the encoder (no full-video buffer)
            feats_b = self.onnx_model.encode_batch(video_b).to(dtype=torch.float32)

            if k == 0:
                frame0 = (feats_b[:, :1], depths_b[:1], intrs_b[:1], extrs_b[:1])
                pcds_f0 = batch_unproject(
                    frame0[1][None], frame0[2][None], frame0[3][None])
                shared_corr_ctx = self.onnx_model.corr_processor.prepare_shared(
                    pcds=pcds_f0, feats=frame0[0], queries=query_point)

            # window A: global [seq_len*k - stride, seq_len*k + stride) —
            # frame0 + carry(stride) + batch[:stride]
            if k > 0 and self.seq_len * k + self.stride <= T:
                feats_w = torch.cat([frame0[0], carry[0], feats_b[:, :self.stride]], 1)
                depths_w = torch.cat([frame0[1], carry[1], depths_b[:self.stride]], 0)
                intrs_w = torch.cat([frame0[2], carry[2], intrs_b[:self.stride]], 0)
                extrs_w = torch.cat([frame0[3], carry[3], extrs_b[:self.stride]], 0)
                self._run_window(feats_w, depths_w, intrs_w, extrs_w,
                                 self.seq_len * k - self.stride,
                                 self.seq_len * k + self.stride, pred,
                                 query_point, query_coords, query_frames,
                                 shared_corr_ctx)
                n_windows += 1

            # window B: global [seq_len*k, seq_len*k + seq_len) — frame0 + full batch
            if self.seq_len * k + self.seq_len <= T:
                feats_w = torch.cat([frame0[0], feats_b], 1)
                depths_w = torch.cat([frame0[1], depths_b], 0)
                intrs_w = torch.cat([frame0[2], intrs_b], 0)
                extrs_w = torch.cat([frame0[3], extrs_b], 0)
                self._run_window(feats_w, depths_w, intrs_w, extrs_w,
                                 self.seq_len * k, self.seq_len * k + self.seq_len,
                                 pred, query_point, query_coords, query_frames,
                                 shared_corr_ctx)
                n_windows += 1

            carry = (feats_b[:, self.stride:], depths_b[self.stride:],
                     intrs_b[self.stride:], extrs_b[self.stride:])
            k += 1

        assert n_windows == len(windows), \
            f"scheduling mismatch: ran {n_windows}, plan says {len(windows)}"
        return pred.coords[0, :total_frames], pred.visibs[0, :total_frames]

    def _run_window(self, feats_w, depths_w, intrs_w, extrs_w, ws, we, pred,
                    query_point, query_coords, query_frames, shared_corr_ctx):
        """Mirror StreamInference._run_window's init logic and call
        onnx_model.forward_window with the 17-frame stack (local bounds 1..17)."""
        seq_len, stride = self.seq_len, self.stride

        coords_init = pred.coords[:, ws:ws + stride]
        visibs_init = pred.visibs[:, ws:ws + stride]
        coords_init = torch.cat(
            [coords_init, repeat(coords_init[:, -1], "b n c -> b w n c", w=stride)], 1)
        visibs_init = torch.cat(
            [visibs_init, repeat(visibs_init[:, -1], "b n -> b w n", w=stride)], 1)
        to_copy = query_frames < we - stride
        coords_init = torch.where(
            repeat(to_copy, "b n -> b w n c", w=seq_len, c=3),
            coords_init,
            repeat(query_coords, "b n c -> b w n c", w=seq_len),
        ).clone()
        visibs_init = torch.where(
            repeat(to_copy, "b n -> b w n", w=seq_len),
            visibs_init,
            torch.zeros_like(visibs_init),
        ).clone()
        track_mask = query_frames < we

        mask = track_mask[0]
        if not mask.any():
            return
        out_coords, out_visibs = self.onnx_model.forward_window(
            feats=feats_w,
            depths=depths_w[None],
            intrs=intrs_w[None],
            extrs=extrs_w[None],
            queries=query_point[:, mask, :],
            coords_init=coords_init[:, :, mask],
            visibs_init=visibs_init[:, :, mask],
            shared_corr_ctx=shared_corr_ctx.select_queries(mask),
            depth_roi=self.depth_roi,
        )
        pred.coords[:, ws:we, mask] = out_coords
        pred.visibs[:, ws:we, mask] = out_visibs
