# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Streaming inference over a Tapip3D_PT2 model.

"""

from typing import Optional, Tuple
import torch
from einops import repeat
from dataclasses import dataclass

from .utils._common import batch_unproject

def plan_windows(total_frames: int, seq_len: int) -> tuple[list[tuple[int, int]], int]:
    """Return (windows, pad): every window (start, end) in execution order.

    Mirrors streaming_forward's padding: pad T to a multiple of seq_len//2.
    """
    stride = seq_len // 2
    pad = (stride - total_frames % stride) % stride
    T = total_frames + pad
    windows = [(we - seq_len, we) for we in range(seq_len, T + 1, stride)]
    return windows, pad

def _require_bool_visibs(visibs: torch.Tensor) -> torch.Tensor:
    visibs = torch.as_tensor(visibs)
    if visibs.dtype != torch.bool:
        # an implicit cast would map any nonzero value to visible — every
        # column would pass and the filter would silently do nothing
        raise TypeError(f"visibs must be bool (thresholded) visibility "
                        f"flags, got dtype {visibs.dtype}")
    return visibs


def filter_visible_tracks(coords: torch.Tensor, visibs: torch.Tensor, *,
                          max_invisible_stems: int = 4,
                          max_reappear_displacement: float = 0.01):
    """Keep the tracked columns that are always visible, or whose every
    invisible run (maximal run of consecutive False stems) is shorter than
    max_invisible_stems — and, when the run is bounded by visible stems on
    both sides, reappears within max_reappear_displacement metres of the
    last visible position. A run touching the trace's first or last stem
    has no reappearance side to measure, so only the length criterion
    applies to it.

    Args:
        coords: (T, Q, 3) world-space trace positions (metres).
        visibs: (T, Q) bool visibility flags (already thresholded by the
            caller's own criterion).

    Returns:
        (keep (Q,) bool, reasons (Q,) list[str | None]) — keep marks the
        surviving columns of the input order; reasons carries one entry
        per column (None for a kept column, the drop reason otherwise).
    """
    coords = torch.as_tensor(coords)
    visibs = _require_bool_visibs(visibs)
    t, q = visibs.shape
    keep = torch.ones(q, dtype=torch.bool, device=visibs.device)
    if t == 0 or q == 0:
        return keep, [None] * q
    reasons: list[str | None] = [None] * q
    for col in range(q):
        if visibs[:, col].all():
            continue
        for a, b in _invisible_runs(visibs[:, col]):
            run_len = b - a + 1
            if run_len >= max_invisible_stems:
                reasons[col] = (f"invisible {run_len} stems "
                                f"(>= {max_invisible_stems})")
                keep[col] = False
                break
            if a > 0 and b < t - 1:  # bounded: reappearance measurable
                d = float(torch.norm(
                    coords[b + 1, col] - coords[a - 1, col]))
                if d >= max_reappear_displacement:
                    reasons[col] = (
                        f"reappears {d:.3f} m away "
                        f"(>= {max_reappear_displacement} m)")
                    keep[col] = False
                    break
    return keep, reasons


def filter_static_tracks(coords: torch.Tensor, visibs: torch.Tensor, *,
                         min_motion_displacement: float = 0.01):
    """Keep only the tracked columns that move: the max displacement from
    the first visible (first-appear) stem to any later visible stem must
    exceed min_motion_displacement metres. Positions of invisible stems
    are the tracker's un-observed extrapolations, so they never count
    toward the displacement. A column without any visible stem has no
    first appearance to measure motion from and is dropped.

    Args:
        coords: (T, Q, 3) world-space trace positions (metres).
        visibs: (T, Q) bool visibility flags (already thresholded by the
            caller's own criterion).

    Returns:
        (keep (Q,) bool, reasons (Q,) list[str | None]) — same convention
        as filter_visible_tracks.
    """
    coords = torch.as_tensor(coords)
    visibs = _require_bool_visibs(visibs)
    t, q = visibs.shape
    keep = torch.ones(q, dtype=torch.bool, device=visibs.device)
    if t == 0 or q == 0:
        return keep, [None] * q
    reasons: list[str | None] = [None] * q
    for col in range(q):
        vis = visibs[:, col]
        if not vis.any():
            reasons[col] = "no visible stem (cannot measure motion)"
            keep[col] = False
            continue
        f = int(torch.nonzero(vis)[0])
        moved = coords[torch.nonzero(vis).flatten(), col] - coords[f, col]
        d = float(torch.norm(moved, dim=1).max())
        if d <= min_motion_displacement:
            reasons[col] = (f"static: max displacement {d:.3f} m "
                            f"(<= {min_motion_displacement} m)")
            keep[col] = False
    return keep, reasons


def _invisible_runs(vis: torch.Tensor) -> list[tuple[int, int]]:
    """(a, b) spans (inclusive) of the maximal consecutive-False runs."""
    runs = []
    a = None
    for i in range(len(vis)):
        if not vis[i] and a is None:
            a = i
        elif vis[i] and a is not None:
            runs.append((a, i - 1))
            a = None
    if a is not None:
        runs.append((a, len(vis) - 1))
    return runs


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

class Tapip3DStreamPT2:
    def __init__(self, pt2_model, queries: torch.Tensor,
                 depth_roi: Optional[torch.Tensor] = None,
                 device: str = "cuda",
                 *, vis_threshold: float = 0.5,
                 max_invisible_stems: int = 4,
                 max_reappear_displacement: float = 0.01,
                 min_motion_displacement: float = 0.01):
        self.pt2_model = pt2_model
        self.queries = queries.to(device)
        assert self.queries.shape[0] == pt2_model.num_queries, (
            f"exact-N contract: queries ({self.queries.shape[0]}) must match "
            f"the iteration graph's num_queries ({pt2_model.num_queries})")
        assert (self.queries[..., 0] < pt2_model.seq_len).all(), (
            "StreamPT2 requires all query frames < seq_len "
            f"({pt2_model.seq_len}) — the static iteration graph cannot "
            "mask late-frame queries")
        self.depth_roi = None if depth_roi is None else depth_roi.to(device)
        self.device = device
        self.seq_len = pt2_model.seq_len
        self.stride = pt2_model.seq_len // 2
        #: filter knobs of run(filter_visible=True) / run(filter_static=True)
        #: — see filter_visible_tracks / filter_static_tracks.
        self.vis_threshold = vis_threshold
        self.max_invisible_stems = max_invisible_stems
        self.max_reappear_displacement = max_reappear_displacement
        self.min_motion_displacement = min_motion_displacement

    @torch.inference_mode()
    def run(self, batches, total_frames: int,
            filter_visible: bool = False, filter_static: bool = False):
        """Run all windows; `batches` yields CPU tuples
        (video (T,3,H,W) in [0,1], depths, intrs, extrs).

        All returned tensors are CPU — the windows run on ``self.device``
        and only the outputs are moved back (callers need no device
        handling).

        With both filters off returns the raw traces of every query column
        as (coords (total_frames, N, 3), visibs (total_frames, N) logits).
        With filter_visible and/or filter_static on, the failing columns
        are dropped instead and the call returns (coords (total_frames,
        N', 3), visibs (total_frames, N') bool at vis_threshold, keep (N,)
        bool, reasons (N,) list[str | None]) — keep/reasons cover the
        original column order (see filter_visible_tracks /
        filter_static_tracks), coords/visibs hold the surviving columns
        only. When both filters are on the static filter runs after the
        visible one: only visible survivors can be dropped as static, and
        a visible drop keeps its own reason. filter_static alone still
        thresholds visibs internally to locate the first-appear stems.

        All query frames must be < seq_len (asserted in __init__);
        late-frame queries would be window-masked, which the static iteration
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

            # ONE batch through the encoder (no full-video buffer). Deviation
            # from the ONNX stream: the PT2 encoder program is static at
            # exactly seq_len frames, so a short batch is padded to seq_len
            # for the graph and the features sliced back to its real length.
            video_enc = video_b
            if video_b.shape[0] < self.seq_len:
                video_enc = torch.cat(
                    [video_b, video_b[-1:].expand(self.seq_len - video_b.shape[0],
                                                  -1, -1, -1)], 0)
            feats_b = self.pt2_model.encode_batch(video_enc)[:, :video_b.shape[0]]

            if k == 0:
                frame0 = (feats_b[:, :1], depths_b[:1], intrs_b[:1], extrs_b[:1])
                pcds_f0 = batch_unproject(
                    frame0[1][None], frame0[2][None], frame0[3][None])
                shared_corr_ctx = self.pt2_model.corr_processor.prepare_shared(
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
        coords = pred.coords[0, :total_frames]
        visibs = pred.visibs[0, :total_frames]
        if not (filter_visible or filter_static):
            return coords.cpu(), visibs.cpu()
        visible = torch.sigmoid(visibs) >= self.vis_threshold
        keep = torch.ones(self.queries.shape[0], dtype=torch.bool,
                          device=coords.device)
        reasons: list[str | None] = [None] * self.queries.shape[0]
        if filter_visible:
            keep, reasons = filter_visible_tracks(
                coords, visible,
                max_invisible_stems=self.max_invisible_stems,
                max_reappear_displacement=self.max_reappear_displacement)
        if filter_static:
            keep_s, reasons_s = filter_static_tracks(
                coords, visible,
                min_motion_displacement=self.min_motion_displacement)
            for col in range(self.queries.shape[0]):
                if keep[col] and not keep_s[col]:
                    keep[col], reasons[col] = False, reasons_s[col]
        return (coords[:, keep].cpu(), visible[:, keep].cpu(),
                keep.cpu(), reasons)

    def _run_window(self, feats_w, depths_w, intrs_w, extrs_w, ws, we, pred,
                    query_point, query_coords, query_frames, shared_corr_ctx):
        """Mirror Tapip3DStreamONNX._run_window's init logic and call
        pt2_model.forward_window with the 17-frame stack and local bounds
        (1, seq_len+1): the PT2 signature is full-video + absolute bounds, so
        the window stack [frame0] + [16 window frames] with bounds (1, 17)
        reproduces the ONNX stream's window semantics exactly."""
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
        out_coords, out_visibs = self.pt2_model.forward_window(
            feats=feats_w,
            depths=depths_w[None],
            intrs=intrs_w[None],
            extrs=extrs_w[None],
            queries=query_point[:, mask, :],
            window_start=1,
            window_end=seq_len + 1,
            coords_init=coords_init[:, :, mask],
            visibs_init=visibs_init[:, :, mask],
            shared_corr_ctx=shared_corr_ctx.select_queries(mask),
            depth_roi=self.depth_roi,
        )
        pred.coords[:, ws:we, mask] = out_coords
        pred.visibs[:, ws:we, mask] = out_visibs
