# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Streaming inference over a Tapip3D_PT2 model.

"""

from typing import Optional
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


def filter_visible_tracks(coords: torch.Tensor, visibs: torch.Tensor,
                          intrs: torch.Tensor, extrs: torch.Tensor, *,
                          max_invisible_stems: int = 8,
                          max_reappear_ratio: float = 3.0):
    """Keep the tracked columns that are always visible, or whose every
    invisible run (maximal run of consecutive False stems) is shorter than
    max_invisible_stems — and, when the run is bounded by visible stems on
    both sides, reappears within a self-relative allowance of the last
    visible position: the pixel gap must stay under ``max_reappear_ratio``
    times the column's expected travel across the gap — its fastest
    per-stem pixel step while continuously visible, times the gap's
    ``run_len + 1`` stems (a mover keeps moving while hidden). A column
    with no such evidence has a zero envelope, so any reappearance gap
    fails it. A run touching the trace's first or last stem has no
    reappearance side to measure, so only the length criterion applies
    to it.

    All displacements are pixel-space. The reappearance gap projects the
    world positions of the last visible stem (a-1) and the reappearance
    stem (b+1) both through the reappearance stem's own pose, so camera
    motion between the stems cancels and only the track's own jump
    remains; the fastest-step envelope is measured the same way (each
    pair of consecutive visible stems under the later stem's pose). The
    self-relative form is what separates a track snap from a genuine
    fast mover: on real Step-2 traces a gripper sweeping ~24 px/stem
    (in a camera-consistent frame) blinks out for 5-7 stems and
    reappears 70-110 px away — consistent with its continued hidden
    travel (~24 px x 6-8 stems), far under the 3x continuation margin,
    while a snapped static column jumps against a ~0px envelope. The
    world-space predecessor of this
    rule (1 cm in metres) read the depth-noise wander of statics (1-7
    cm) and the real travel of briefly-occluded movers alike as
    failures and dropped the movers this pipeline wants to keep.

    Args:
        coords: (T, Q, 3) world-space trace positions (metres).
        visibs: (T, Q) bool visibility flags (already thresholded by the
            caller's own criterion).
        intrs: (T, 3, 3) intrinsics of the tracked camera per stem.
        extrs: (T, 4, 4) w2c extrinsics of the tracked camera per stem.
        max_reappear_ratio: margin over the column's expected travel
            across the gap (fastest per-stem visible step x the gap's
            stems); a drop needs the gap at this many times the expected
            continuation, or more (default 3).

    Returns:
        (keep (Q,) bool, reasons (Q,) list[str | None]) — keep marks the
        surviving columns of the input order; reasons carries one entry
        per column (None for a kept column, the drop reason otherwise).
    """
    coords = torch.as_tensor(coords)
    visibs = _require_bool_visibs(visibs)
    t, q = visibs.shape
    intrs = torch.as_tensor(intrs, device=coords.device, dtype=coords.dtype)
    extrs = torch.as_tensor(extrs, device=coords.device, dtype=coords.dtype)
    if intrs.shape != (t, 3, 3) or extrs.shape != (t, 4, 4):
        raise ValueError(
            f"intrinsics/extrinsics must be ({t}, 3, 3) / ({t}, 4, 4), got "
            f"{tuple(intrs.shape)} / {tuple(extrs.shape)}")
    keep = torch.ones(q, dtype=torch.bool, device=visibs.device)
    if t == 0 or q == 0:
        return keep, [None] * q
    ones = coords.new_ones(2, 1)

    # per-column fastest per-stem pixel step over consecutive visible
    # stems (both positions under the later stem's pose)
    fastest = torch.zeros(q, dtype=coords.dtype, device=coords.device)
    for i in range(t - 1):
        pair = visibs[i] & visibs[i + 1]
        if not pair.any():
            continue
        idx = torch.nonzero(pair).flatten()
        h = torch.cat([coords[i, idx], ones[:1].expand(idx.numel(), 1)],
                      dim=-1)
        h2 = torch.cat([coords[i + 1, idx],
                        ones[:1].expand(idx.numel(), 1)], dim=-1)
        cam0 = (extrs[i + 1] @ h.t()).t()[:, :3]
        cam1 = (extrs[i + 1] @ h2.t()).t()[:, :3]
        z0, z1 = cam0[:, 2], cam1[:, 2]
        ok = (z0 > 0) & (z1 > 0)
        if not ok.any():
            continue
        img0 = intrs[i + 1] @ (cam0[ok] / z0[ok, None]).t()
        img1 = intrs[i + 1] @ (cam1[ok] / z1[ok, None]).t()
        p0 = (img0[:2] / img0[2:3]).t()
        p1 = (img1[:2] / img1[2:3]).t()
        step = torch.norm(p1 - p0, dim=1)
        m = fastest[idx[ok]]
        fastest[idx[ok]] = torch.maximum(m, step)

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
                # the point keeps moving while hidden: its expected travel
                # across the gap is its fastest visible per-stem step times
                # the gap's stems; a snap must exceed the margin over that
                allowance = (max_reappear_ratio * float(fastest[col])
                             * (run_len + 1))
                # both world positions through the reappearance stem's own
                # pose (camera motion between the stems cancels)
                h = torch.cat([coords[[a - 1, b + 1], col], ones], dim=-1)
                cam = (extrs[b + 1] @ h.t()).t()[:, :3]
                z = cam[:, 2]
                if (z > 0).all():
                    img = intrs[b + 1] @ (cam / z[:, None]).t()
                    px = (img[:2] / img[2:3]).t()
                    d = float(torch.norm(px[1] - px[0]))
                    if d >= allowance:
                        reasons[col] = (
                            f"reappears {d:.1f} px away "
                            f"(>= {allowance:.1f} px)")
                        keep[col] = False
                        break
    return keep, reasons


def _project_pixels(coords: torch.Tensor, intrs: torch.Tensor,
                    extrs: torch.Tensor) -> torch.Tensor:
    """(T, Q, 2) pixels of the world coords under each stem's own
    intrinsics and w2c extrinsics — nan on a stem whose projection falls
    behind the camera (z <= 0)."""
    t, q, _ = coords.shape
    px = torch.full((t, q, 2), float("nan"), dtype=coords.dtype,
                    device=coords.device)
    ones = coords.new_ones(q, 1)
    for i in range(t):
        h = torch.cat([coords[i], ones], dim=-1)       # (Q, 4)
        cam = (extrs[i] @ h.t()).t()[:, :3]            # (Q, 3) w2c
        z = cam[:, 2:3]
        good = z[:, 0] > 0
        if not good.any():
            continue
        cam = cam[good] / z[good]
        img = (intrs[i] @ cam.t()).t()                 # (n, 3)
        px[i, good] = img[:, :2] / img[:, 2:3]
    return px


def filter_static_pixel_tracks(coords: torch.Tensor, visibs: torch.Tensor,
                               intrs: torch.Tensor, extrs: torch.Tensor, *,
                               min_motion_pixels: float = 3.0):
    """Keep only the tracked columns that move in the camera's pixels:
    the max displacement of the column's reprojection from the pixel of
    its first visible (first-appear) stem to any later visible stem must
    exceed min_motion_pixels. The pixel-space criterion is robust to
    depth noise and pose drift: world-coordinate wander along the
    viewing ray or a whole-scene Step-2 coordinate jump move a static
    point's reprojection by far less than its world position (each stem
    projects through that stem's own pose, so accumulated frame drift
    does not accumulate in pixels).

    Args:
        coords: (T, Q, 3) world-space trace positions (metres).
        visibs: (T, Q) bool visibility flags (already thresholded by the
            caller's own criterion).
        intrs: (T, 3, 3) intrinsics of the tracked camera per stem.
        extrs: (T, 4, 4) w2c extrinsics of the tracked camera per stem.

    Returns:
        (keep (Q,) bool, reasons (Q,) list[str | None]) — same convention
        as filter_visible_tracks.
    """
    coords = torch.as_tensor(coords)
    visibs = _require_bool_visibs(visibs)
    t, q = visibs.shape
    intrs = torch.as_tensor(intrs, device=coords.device, dtype=coords.dtype)
    extrs = torch.as_tensor(extrs, device=coords.device, dtype=coords.dtype)
    if intrs.shape != (t, 3, 3) or extrs.shape != (t, 4, 4):
        raise ValueError(
            f"intrinsics/extrinsics must be ({t}, 3, 3) / ({t}, 4, 4), got "
            f"{tuple(intrs.shape)} / {tuple(extrs.shape)}")
    keep = torch.ones(q, dtype=torch.bool, device=visibs.device)
    if t == 0 or q == 0:
        return keep, [None] * q
    px = _project_pixels(coords, intrs, extrs)
    reasons: list[str | None] = [None] * q
    for col in range(q):
        vis = visibs[:, col]
        if not vis.any():
            reasons[col] = "no visible stem (cannot measure motion)"
            keep[col] = False
            continue
        idx = torch.nonzero(vis).flatten()
        f = int(idx[0])
        d = torch.norm(px[idx, col] - px[f, col], dim=1)
        d = d[torch.isfinite(d)]
        if d.numel() == 0:
            reasons[col] = ("no projectable visible stem (behind the "
                            "camera / missing geometry)")
            keep[col] = False
            continue
        dmax = float(d.max())
        if dmax <= min_motion_pixels:
            reasons[col] = (f"static (pixel): max displacement {dmax:.1f} "
                            f"px (<= {min_motion_pixels:.1f} px)")
            keep[col] = False
    return keep, reasons


def _merge_drops(keep: torch.Tensor, reasons: list,
                 keep_new: torch.Tensor, reasons_new: list) -> None:
    """Fold a later filter's drops into keep/reasons: a column dropped by
    the earlier filter keeps its reason, only new drops adopt the later
    filter's reason (in place)."""
    for col in range(keep.numel()):
        if keep[col] and not keep_new[col]:
            keep[col], reasons[col] = False, reasons_new[col]


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
                 max_invisible_stems: int = 8,
                 max_reappear_ratio: float = 3.0,
                 min_motion_pixels: float = 5.0):
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
        #: filter knobs of run(filter_visible=True) /
        #: run(filter_static_pixel=True) — see filter_visible_tracks /
        #: filter_static_pixel_tracks. max_reappear_ratio is the
        #: visible filter's self-relative reappearance tolerance
        #: (world-space displacement proved too strict:
        #: briefly-occluded movers read as track failures).
        self.vis_threshold = vis_threshold
        self.max_invisible_stems = max_invisible_stems
        self.max_reappear_ratio = max_reappear_ratio
        self.min_motion_pixels = min_motion_pixels

    @torch.inference_mode()
    def run(self, batches, total_frames: int,
            filter_visible: bool = False,
            filter_static_pixel: bool = False):
        """Run all windows; `batches` yields CPU tuples
        (video (T,3,H,W) in [0,1], depths, intrs, extrs).

        All returned tensors are CPU — the windows run on ``self.device``
        and only the outputs are moved back (callers need no device
        handling).

        With all filters off returns the raw traces of every query column
        as (coords (total_frames, N, 3), visibs (total_frames, N) logits).
        With filter_visible and/or filter_static_pixel on, the failing
        columns are dropped instead and the call returns (coords
        (total_frames, N', 3), visibs (total_frames, N') bool at
        vis_threshold, keep (N,) bool, reasons (N,) list[str | None]) —
        keep/reasons cover the original column order (see
        filter_visible_tracks / filter_static_pixel_tracks),
        coords/visibs hold the surviving columns only. The pixel filter
        runs after the visible one, narrowing the survivors further:
        only visible survivors can be dropped as static, and an earlier
        drop keeps its own reason. A static filter alone still
        thresholds visibs internally to locate the first-appear stems.
        filter_static_pixel measures the reprojected pixels against each
        stem's own intrinsics/extrinsics (collected from the batches),
        immune to depth noise (see filter_static_pixel_tracks).

        All query frames must be < seq_len (asserted in __init__);
        late-frame queries would be window-masked, which the static iteration
        graph cannot represent."""
        # the visible filter measures reappearances in pixels and the
        # pixel static filter in reprojected pixels: both need the
        # per-stem cameras of the tracked camera
        geoms = [] if (filter_visible or filter_static_pixel) else None
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
            if geoms is not None:
                geoms.append((intrs_b, extrs_b))
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
        if not (filter_visible or filter_static_pixel):
            return coords.cpu(), visibs.cpu()
        visible = torch.sigmoid(visibs) >= self.vis_threshold
        keep = torch.ones(self.queries.shape[0], dtype=torch.bool,
                          device=coords.device)
        reasons: list[str | None] = [None] * self.queries.shape[0]
        if filter_visible or filter_static_pixel:
            # per-stem geometry of the tracked camera (the batch loop
            # collects the padded batches: exactly T stems, trimmed to the
            # unpadded trace length) — the pixel criteria' reference
            intrs = torch.cat([g[0] for g in geoms])[:total_frames] \
                if geoms else torch.empty(0, 3, 3)
            extrs = torch.cat([g[1] for g in geoms])[:total_frames] \
                if geoms else torch.empty(0, 4, 4)
        if filter_visible:
            keep, reasons = filter_visible_tracks(
                coords, visible, intrs, extrs,
                max_invisible_stems=self.max_invisible_stems,
                max_reappear_ratio=self.max_reappear_ratio)
        if filter_static_pixel:
            keep_p, reasons_p = filter_static_pixel_tracks(
                coords, visible, intrs, extrs,
                min_motion_pixels=self.min_motion_pixels)
            _merge_drops(keep, reasons, keep_p, reasons_p)
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
