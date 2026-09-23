"""MoGe v3 monocular metric depth over the episodes of a LeRobot dataset.

Sibling to ``infer_moge3.py``.  That tool takes a folder of images and writes
one artifact per image; this one takes a dataset of *episodes* and writes one
lossless FFV1 video per episode-camera, each frame packed the same way the
image tool packs a still — ``[Oct_U, Oct_V, depth]`` across an RGB uint8, read
back with ``NormalDepthPack``.  The dataset is reached through
:class:`utils.libero_wrapper.LiberoWrapper`, which is the only way to get
pixels out of these videos: they are AV1, and the local OpenCV opens them but
never returns a frame.

Three things drive most of what is below:

* **A video needs one depth range, not one per frame.**  The codec's
  ``resolve_range`` tightens the rails to whatever the frame in hand actually
  uses, which is the right call for a still and the wrong one for a video: the
  depth channel's brightness would shift frame to frame, so a static scene
  would flicker, and no single sidecar could describe the result.  So a few
  frames are sampled up front, inferred, and their ranges unioned into one
  range the whole episode is encoded over.

* **Monocular depth is only defined up to a similarity, and the model refits it
  every frame.**  Left alone, the same static scene comes back at a slightly
  different scale from each frame, which reads as the geometry breathing.
  ``--align`` registers every frame back to the episode's first with a RANSAC
  similarity (``utils.visualize.moge_register``) and re-renders it into that
  frame's camera, so the episode's geometry is consistent against one
  reference.  That only means anything for a *fixed* camera, so it is refused
  by name on a wrist camera and refused at the episode level when the sampled
  frames do not register — see ``_survey_episode``.

  **It is off by default, and that is a watching-the-output decision.**  On
  every measure taken here it improves: the background's frame-to-frame spread
  falls 81%, central reprojection holes are 0.0000%, and the registrations are
  good (7/7 sampled frames, scale 0.97-1.52 at 45-73% inliers).  It still reads
  as *shakier* than the frames as the model gave them, because it resamples
  every frame through a per-frame registration whose own fit jitters — and the
  depth spread those numbers come from is not what the eye tracks.  Treat the
  numbers as necessary and not sufficient; the video is the test.

* **The anchor's own fit becomes the episode's one scale.**  Aligned depth is
  metric, in the anchor's frame; it is converted to the codec's relative depth
  through the anchor frame's ``(shift, metric_scale)``.  So the sidecar carries
  one *scalar* pair where an unaligned run carries per-frame arrays — which is
  the point, since a per-frame pair would modulate the packed brightness by up
  to 40% (measured 0.77-1.08 on one clip) and reintroduce the flicker the
  alignment just removed.  ``aligned`` in the sidecar says which shape to
  expect.
"""

import argparse
import contextlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from depth_models.moge3.moge_pt2 import MoGev3_PT2
from utils.file_io.video_io import FFV1VideoWriter, H264VideoWriter
from utils.libero_wrapper import Episode, LiberoWrapper
from utils.normal_depth_pack import (
    DEFAULT_Z_MAX,
    DEFAULT_Z_MIN,
    MOGE_POLE,
    DepthScale,
    NormalDepthPack,
)
from utils.visualize.moge_register import (
    Registration,
    register_to_anchor,
    reproject_to_camera,
)

#: The export this tool is written against.  ``input_size`` must match the
#: checkpoint's export resolution or the graph's shape check rejects the call,
#: so the two defaults below travel together.
DEFAULT_PT2 = "weights/moge3/moge3_l_512x512.pt2"
DEFAULT_INPUT_SIZE = (512, 512)

#: How many frames to infer when picking the episode's depth range.  The range
#: only has to be an upper bound -- a frame that overshoots it is clipped by
#: the encoder, not rejected -- so a sparse sample is safe, and eight frames
#: out of a hundred-odd costs about 6% of the run.  These are also the frames
#: the fixed-camera check is made on, which is why they are spread across the
#: whole episode rather than taken from the front.
DEFAULT_RANGE_SAMPLE_FRAMES = 8

#: Per-frame gate: the fraction of sampled correspondences that must agree
#: before a frame is aligned.  Deliberately far below the 87-98% that adjacent
#: frames of a fixed camera read, because the model's geometry drifts and the
#: drift is the thing being corrected: a frame registered against an anchor 40
#: frames back reads 45-73% however static the scene is, and the further back
#: the lower.  See ``register_to_anchor`` for the measurements.
#:
#: The number is load-bearing, and it was set too high at first.  On one
#: 138-frame LIBERO episode a gate of 0.55 aligned 111 frames and a gate of
#: 0.40 aligned 135; the frames in between were correct registrations being
#: thrown away, and they cost the episode most of its temporal consistency --
#: the background's frame-to-frame spread fell 22% at 0.55 and 81% at 0.40.
DEFAULT_MIN_INLIERS = 0.40

#: Per-episode backstop: the fraction of the sampled frames that must clear
#: ``--min-inliers`` before the episode is accepted as a fixed camera.  This is
#: not how a wrist camera is caught -- `parse_args` refuses `--align` on one by
#: name, which is knowable before a frame is decoded and cannot be fooled by a
#: clip where the arm holds still.  It is here for a camera that *is* moving
#: under another name, where the name check cannot help.
#:
#: Measured at the 0.40 gate over four episodes each: the fixed camera
#: registered 7/7, 7/7, 7/7 and 6/7, the wrist camera 0/7 every time.  The
#: separation is wide because the *long-gap* ratios are what this counts, and
#: a moving camera does not hold a fit across one at all.
DEFAULT_MIN_REGISTERED_FRAC = 0.5

#: Quality/size dial for `--save-viz`.  x264's own default: higher is smaller
#: and worse, and this file exists to be looked at, not measured.
DEFAULT_VIZ_CRF = 23

#: Substring that marks a camera as arm-mounted, and so as one that `--align`
#: cannot work on.  A wrist camera's correspondence is the pixel grid like any
#: other, but it is riding the arm: the pixel grid stops describing the same
#: scene between frames, and the registration goes on returning a plausible
#: transform regardless.
WRIST_CAMERA_MARKER = "wrist"


class FramePrediction(NamedTuple):
    """One frame's model output, reduced to what the packer and aligner need.

    ``depth_m`` is the metric depth in the frame's own camera and ``points`` is
    the matching camera-space point map, so ``points[..., 2] == depth_m`` where
    the pixel is valid.  Both are zeroed (``points`` to NaN) where the model
    had nothing to say.  ``intrinsics`` is MoGe's (3, 3) matrix in *normalized*
    uv units — ``fx = k[0,0]``, ``cx = k[0,2] = 0.5``, pixel centres at
    ``(j + 0.5) / W`` — which is what ``reproject_to_camera`` expects.
    """

    depth_m: np.ndarray  # (H, W) float32, metres
    normal: np.ndarray  # (H, W, 3) float32, unit
    valid: np.ndarray  # (H, W) bool
    points: np.ndarray  # (H, W, 3) float32, camera-space metres
    intrinsics: np.ndarray  # (3, 3) float64, normalized uv
    shift: float  # the model's per-frame metric fit
    metric_scale: float


class Anchor(NamedTuple):
    """The episode frame every other frame is registered to.

    The first kept frame, and therefore the world frame the whole episode is
    expressed in.  It is never itself registered — a frame against itself is
    the identity — so it passes through ``frame_geometry`` untouched.
    """

    index: int  # absolute dataset index
    prediction: FramePrediction

    def scale(self) -> DepthScale:
        """The model's own fit for the anchor frame.

        This becomes the *episode's* relative-depth basis when aligning, which
        is what keeps the packed channel stable: one pair for the whole video
        instead of a pair per frame.
        """
        return DepthScale(shift=self.prediction.shift,
                          metric_scale=self.prediction.metric_scale)


class Survey(NamedTuple):
    """What pass 1 decided about an episode, carried to the encode loop."""

    z_min: float
    z_max: float
    anchor: Anchor | None  # None when --no-align
    registered: int  # sampled frames that cleared --min-inliers
    considered: int  # sampled frames that were registered at all


# --------------------------------------------------------------- sampling

def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """*k* frame indices spread evenly across ``[0, n_frames)``, ends included.

    The indices are distinct by construction, and the early return is what
    makes them so rather than just an optimisation: it leaves ``k < n_frames``
    for the linspace, so consecutive samples are more than one frame apart, and
    two numbers further apart than 1 cannot round to the same integer.  Asking
    for eight samples of a three-frame clip therefore takes the early return
    rather than inferring the middle frame several times over.
    """
    if n_frames <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    return sorted(int(round(i)) for i in np.linspace(0.0, n_frames - 1, k))


def resolve_video_range(
    pack: NormalDepthPack, frames: Sequence[tuple[np.ndarray, np.ndarray | None]]
) -> tuple[float, float]:
    """One ``(z_min, z_max)`` for a whole episode, from a sample of its frames.

    The union of what the sampled frames use, clipped to the pack's rails --
    ``pack.resolve_range`` applied to the video rather than to a frame.  A
    frame that runs past the result is clipped by the encoder rather than
    refused, which is what makes a sparse sample safe.

    Frames with nothing representable contribute nothing rather than dragging
    in the rails, so one unusable sample does not cost the whole video its
    adaptivity.  Falls back to the rails when no sample is usable, or when the
    union collapses to a point, since a zero span cannot be divided by.
    """
    lo: float | None = None
    hi: float | None = None
    for depth_z, valid in frames:
        usable = pack.valid_mask(depth_z, valid)
        if not usable.any():
            continue
        frame_lo = float(depth_z[usable].min())
        frame_hi = float(depth_z[usable].max())
        lo = frame_lo if lo is None else min(lo, frame_lo)
        hi = frame_hi if hi is None else max(hi, frame_hi)

    if lo is None:
        return pack.z_min, pack.z_max
    lo = max(pack.z_min, lo)
    hi = min(pack.z_max, hi)
    if not lo < hi:
        return pack.z_min, pack.z_max
    return lo, hi


# ------------------------------------------------------------------ model

def infer_frame(model: MoGev3_PT2, frame_rgb: np.ndarray) -> FramePrediction:
    """Run one RGB frame through the model and reduce it to packing terms.

    Mirrors the postprocess in ``infer_moge3.py`` -- see this module's
    docstring for why it is copied rather than shared.  The model masks by
    writing ``inf`` into ``depth`` and ``points``, so validity is read off the
    finite values rather than taken from ``mask``, which is the same test the
    rest of the pipeline would have to make anyway.

    Points are NaN where invalid, not zero: a zero point is a real point at the
    camera centre, and the registration and the splat both skip non-finite
    geometry, so NaN is what makes "no data here" survive those paths.
    """
    out = model.infer(frame_rgb)
    depth_raw = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)
    points_raw = out["points"].squeeze(0).cpu().numpy().astype(np.float32)
    valid = (np.isfinite(depth_raw) & (depth_raw > 0.0)
             & np.isfinite(points_raw).all(axis=-1))
    return FramePrediction(
        depth_m=np.where(valid, depth_raw, 0.0).astype(np.float32),
        normal=out["normal"].squeeze(0).cpu().numpy().astype(np.float32),
        valid=valid,
        points=np.where(valid[..., None], points_raw, np.nan).astype(np.float32),
        intrinsics=out["intrinsics"].squeeze(0).cpu().numpy().astype(np.float64),
        shift=float(out["shift"].reshape(-1)[0]),
        metric_scale=float(out["metric_scale"].reshape(-1)[0]),
    )


def frame_geometry(
    prediction: FramePrediction,
    anchor: FramePrediction,
    registration: Registration | None,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The ``(depth_m, normals, valid, hole_fraction)`` a frame is packed from.

    With no ``registration`` the frame is packed as it came out of the model,
    in its own camera -- which is the whole of what ``--no-align`` does, and
    also the fallback for a frame whose registration was rejected.

    A rejected frame still has to be *decodable*, though: the sidecar carries
    one metric basis for episodes that were aligned, so a frame packed in its
    own camera must be expressed in that basis rather than its own.  Since the
    basis is a per-video affine and the depth is metric, that is exactly the
    conversion the caller does anyway -- the only thing a rejected frame gives
    up is the rotation of its normals and the resampling of its pixels.

    A rejection is not a small thing to give up, and it is worth being clear
    that it is not "the transform was near the identity anyway": on a fixed
    camera a rejection means the inlier gate failed, which is what the model's
    *largest* drift looks like.  The frame is still packed, decodably and in
    the right basis, and the refusal is recorded -- but its geometry is the
    unaligned geometry.

    The hole fraction is NaN rather than 0.0 here: no re-render happened, so
    there is no hole count to report, and a zero would read as "this frame
    re-rendered perfectly" to anything that averages the column.
    """
    if registration is None:
        return prediction.depth_m, prediction.normal, prediction.valid, np.nan
    return reproject_to_camera(
        prediction.points,
        prediction.normal,
        prediction.valid,
        registration.scale,
        registration.rotation,
        registration.translation,
        anchor.intrinsics,
        size,
        # The frame's points were unprojected through *its* recovered focal, and
        # MoGe refits that every frame (measured 1.09-1.19 across one episode).
        # Finding which source pixel a point came from needs the focal it was
        # built with, not the anchor's.
        source_intrinsics=prediction.intrinsics,
    )


def encoded_scale(
    anchor: Anchor | None, prediction: FramePrediction
) -> DepthScale:
    """The affine a frame's metric depth is packed through.

    One basis for the whole episode when aligning (the anchor's own fit, which
    is what keeps the packed channel stable), each frame's own fit when not.

    Both the range survey and the encode loop go through this, and they have to
    agree: the survey measures the range over the depths the *encoder* will
    pack, so measuring it on the raw metric depth instead -- as it once did --
    widens the range by that frame's ``metric_scale`` and spends codes on
    nothing.
    """
    if anchor is not None:
        return anchor.scale()
    return DepthScale(shift=prediction.shift,
                      metric_scale=prediction.metric_scale)


def _register(
    prediction: FramePrediction,
    anchor: Anchor,
    min_inliers: float,
) -> Registration | None:
    """Register a frame to the episode's anchor, or ``None`` if it is refused."""
    return register_to_anchor(
        prediction.points,
        prediction.depth_m,
        anchor.prediction.points,
        anchor.prediction.depth_m,
        min_inliers=min_inliers,
    )


# --------------------------------------------------------------- episode io

def _kept_frames(episode: Episode, args: argparse.Namespace) -> list[int]:
    """The absolute dataset indices this run keeps, in order."""
    kept = list(range(episode.from_index, episode.to_index, args.stride))
    if args.max_frames:
        kept = kept[: args.max_frames]
    if not kept:
        raise RuntimeError("no frames selected")
    return kept


def _survey_episode(
    model: MoGev3_PT2,
    pack: NormalDepthPack,
    wrapper: LiberoWrapper,
    episode: Episode,
    kept: Sequence[int],
    args: argparse.Namespace,
) -> Survey:
    """Fix the episode's ``(z_min, z_max)``, build its anchor, and check the
    camera is fixed.

    The frames are inferred at ``--range-sample-frames`` spread across the
    episode, so the range covers its whole span rather than its opening, and
    the range is measured on the depths that are actually *encoded* -- the
    similarity rescales them when aligning, and each frame's own affine fit
    rescales them when not, so a range taken on the raw metric depth would be
    the wrong one either way.

    The anchor is the first kept frame, which is always in the sample, so it
    comes out of this pass rather than costing a second inference of the same
    frame.

    Raises when the camera is not fixed.  Alignment is only defined for one
    (the correspondence is the pixel grid itself), and a moving camera does not
    merely degrade it: the RANSAC still returns a plausible-looking
    similarity, so nothing downstream can tell the result is wrong.  Refusing
    is the only way the failure stays visible.
    """
    sample = sample_frame_indices(len(kept), args.range_sample_frames)
    indices = sorted({kept[0]} | {kept[i] for i in sample})

    encoded: list[tuple[np.ndarray, np.ndarray | None]] = []
    anchor = None
    registered = considered = 0
    for index, frame in wrapper.frames(episode, indices):
        prediction = infer_frame(model, frame)
        registration = None
        if args.align and index == kept[0]:
            anchor = Anchor(index=index, prediction=prediction)
        elif anchor is not None:
            registration = _register(prediction, anchor, args.min_inliers)
            considered += 1
            registered += registration is not None

        depth_m, _, valid, _ = frame_geometry(
            prediction, anchor.prediction if anchor else prediction,
            registration, (model.height, model.width),
        )
        encoded.append((encoded_scale(anchor, prediction).from_metric(depth_m),
                        valid))

    if anchor is not None and considered and registered / considered < args.min_registered_frac:
        raise RuntimeError(
            f"only {registered} of {considered} sampled frames registered "
            f"against the episode's first frame, so {wrapper.camera!r} does "
            f"not look like a fixed camera. Alignment needs one: its "
            f"correspondence is the pixel grid, which a moving camera "
            f"invalidates, and the registration would go on returning a "
            f"plausible-looking transform for it. Re-run with --no-align to "
            f"process this camera unaligned."
        )

    z_min, z_max = resolve_video_range(pack, encoded)
    return Survey(z_min, z_max, anchor, registered, considered)


def process_episode(
    model: MoGev3_PT2,
    pack: NormalDepthPack,
    wrapper: LiberoWrapper,
    episode: Episode,
    mkv_path: Path,
    npz_path: Path,
    args: argparse.Namespace,
) -> tuple[int, int, int]:
    """Encode one episode-camera.

    Returns ``(frames written, sampled frames registered, sampled frames
    considered)``; the last two are zero when alignment is off.

    Both outputs are written to a ``.part`` name and moved into place only once
    ffmpeg has exited cleanly, so a run killed mid-episode leaves nothing that
    ``--skip-existing`` would mistake for finished work.
    """
    kept = _kept_frames(episode, args)
    # Two spellings, because the two APIs disagree and both are silent about it
    # on a square frame: the writer takes (width, height) while the reprojection
    # and the codec's sidecar take (height, width). Naming them apart is what
    # stops one being handed the other's argument -- which is exactly what a
    # single `out_size` did, and it only shows up on a non-square export.
    writer_size = (model.width, model.height)
    frame_size = (model.height, model.width)

    survey = _survey_episode(model, pack, wrapper, episode, kept, args)
    z_min, z_max = survey.z_min, survey.z_max
    anchor = survey.anchor
    # The pair the sidecar is built with. Unaligned it is written and then
    # overwritten by the per-frame arrays below, so the identity is only a
    # placeholder for a slot that never survives.
    scale = anchor.scale() if anchor else DepthScale()

    mkv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mkv = mkv_path.with_name(mkv_path.name + ".part")
    tmp_npz = npz_path.with_name(npz_path.name + ".part")
    viz_path = plan_viz_path(mkv_path) if args.save_viz else None
    tmp_viz = viz_path.with_name(viz_path.name + ".part") if viz_path else None

    # Only one of the two sets is ever written, so only one is collected: an
    # aligned episode has no use for the per-frame fits and an unaligned one
    # has no transforms to record.
    shifts: list[float] = []
    scales: list[float] = []
    align_scales: list[float] = []
    align_rotations: list[np.ndarray] = []
    align_translations: list[np.ndarray] = []
    align_inliers: list[float] = []
    align_holes: list[float] = []
    # The two writers are opened together and closed together, so a failure in
    # either leaves neither file behind.  nullcontext keeps the visualization
    # from splitting the loop into two copies.
    viz = (H264VideoWriter(str(tmp_viz), fps=wrapper.fps / args.stride,
                           size=(2 * writer_size[0], writer_size[1]),
                           threads=args.threads, crf=args.viz_crf)
           if tmp_viz is not None else contextlib.nullcontext())
    try:
        with FFV1VideoWriter(
            str(tmp_mkv),
            fps=wrapper.fps / args.stride,
            size=writer_size,
            threads=args.threads,
        ) as writer, viz as viz_writer:
            for index, frame in wrapper.frames(episode, kept):
                prediction = infer_frame(model, frame)
                registration = None
                if anchor is not None and index != anchor.index:
                    registration = _register(prediction, anchor, args.min_inliers)

                depth_m, normals, valid, holes = frame_geometry(
                    prediction, anchor.prediction if anchor else prediction,
                    registration, frame_size,
                )
                plate = pack.encode(normals, encoded_scale(anchor, prediction)
                                    .from_metric(depth_m), valid,
                                    z_range=(z_min, z_max))
                writer.write_frame(plate)
                if viz_path is not None:
                    # The prompt's own frame beside what it produced, so the
                    # result can be judged against what the model was given
                    # rather than from memory.
                    viz_writer.write_frame(
                        np.concatenate([_fit_to(frame, frame_size), plate], axis=1))

                if anchor is None:
                    shifts.append(prediction.shift)
                    scales.append(prediction.metric_scale)
                else:
                    # A refused frame records NaN rather than the identity, so
                    # a reader averaging these cannot mistake "packed as it
                    # came out of the model" for a perfect fit.
                    applied = registration is not None
                    align_scales.append(registration.scale if applied else np.nan)
                    align_rotations.append(registration.rotation if applied
                                           else np.full((3, 3), np.nan))
                    align_translations.append(registration.translation if applied
                                              else np.full(3, np.nan))
                    align_inliers.append(registration.inliers if applied else np.nan)
                    align_holes.append(holes)
            written = writer.frame_count

        sidecar = pack.scale_dict(
            scale,
            z_min,
            z_max,
            frame_size,
            fps=np.float64(wrapper.fps / args.stride),
            n_frames=np.int32(written),
            dataset=np.asarray(wrapper.name),
            camera=np.asarray(wrapper.camera),
            episode=np.int32(episode.index),
            task=np.asarray(episode.task),
            stride=np.int32(args.stride),
            range_sample_frames=np.int32(len(
                sample_frame_indices(len(kept), args.range_sample_frames))),
            aligned=np.bool_(anchor is not None),
        )
        if anchor is not None:
            sidecar["anchor_frame"] = np.int64(anchor.index)
            sidecar["align_scale"] = np.asarray(align_scales, dtype=np.float64)
            sidecar["align_R"] = np.stack(align_rotations).astype(np.float64)
            sidecar["align_t"] = np.stack(align_translations).astype(np.float64)
            sidecar["align_inliers"] = np.asarray(align_inliers, dtype=np.float64)
            sidecar["align_holes"] = np.asarray(align_holes, dtype=np.float64)
        else:
            # The codec's sidecar is built for a still, where one shift and one
            # metric_scale describe the whole frame.  Unaligned, a video's are
            # fitted per frame, so the scalars these two would be become
            # arrays -- same keys, so a reader that wants the ordinary keys
            # (z_min/z_max/pole) still finds them where it expects.
            sidecar["shift"] = np.asarray(shifts, dtype=np.float64)
            sidecar["metric_scale"] = np.asarray(scales, dtype=np.float64)

        # A file handle, not a path: np.savez appends ".npz" to any string path
        # that does not already end in it, so the ".part" temp name would be
        # written as "<name>.npz.part.npz" and the rename below would find
        # nothing there.
        with open(tmp_npz, "wb") as handle:
            np.savez(handle, **sidecar)

        # Every file is complete before any is moved, so a failure above
        # leaves the set absent rather than half-present.
        os.replace(tmp_mkv, mkv_path)
        os.replace(tmp_npz, npz_path)
        if viz_path is not None:
            os.replace(tmp_viz, viz_path)
    except BaseException:
        # Leave nothing half-written behind for the next run to trip over.
        for leftover in (tmp_mkv, tmp_npz, tmp_viz):
            if leftover is not None:
                leftover.unlink(missing_ok=True)
        raise
    return written, survey.registered, survey.considered


# -------------------------------------------------------------------- cli

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoGe v3 monocular metric depth over the episodes of a "
        "LeRobot dataset -> one octahedral-packed lossless FFV1 video per "
        "episode-camera, plus a .npz sidecar beside it.",
    )
    parser.add_argument(
        "--repo-id", required=True,
        help="Dataset id, passed to the LeRobot constructors. Nothing is "
             "fetched from the Hub, so any stable id works as long as it is "
             "the same one throughout.",
    )
    parser.add_argument(
        "--data-root", required=True,
        help="The dataset directory itself (holding meta/, data/ and videos/) "
             "-- e.g. data/libero_mujoco3.3.2/lerobot_v30/libero_goal_no_noops_lerobot. "
             "Only LeRobot v3.0 copies are readable.",
    )
    parser.add_argument(
        "--camera", default="observation.images.image",
        help="Camera key to decode (default %(default)s). Alignment is only "
             "meaningful for a fixed camera; a wrist camera must be run with "
             "--no-align.",
    )
    parser.add_argument(
        "--out-dir", default="./output/moge3_video",
        help="Output root; each episode lands at "
             "<out-dir>/<dataset>/<camera>/episode_%06d.mkv.",
    )
    parser.add_argument(
        "--episode-idxes", nargs="*", type=int, default=None,
        help="Process only these episode indices (default: all of them).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N episodes; default: all of them.",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth frame. The output is written at fps/N so it still "
             "plays at the source's speed (default 1).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the frames taken from each episode; default: all of them.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip an episode whose .mkv and .npz are both already written. "
             "Safe to resume with: an episode is only in place once ffmpeg "
             "exited cleanly.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be processed and the raw output size, without "
             "loading the model or writing anything.",
    )
    # BooleanOptionalAction declares --align and --no-align together; a
    # mutually-exclusive group would spell out both and then let the default
    # contradict one of them.
    parser.add_argument(
        "--align", action=argparse.BooleanOptionalAction, default=False,
        help="Register every frame back to the episode's first frame and "
             "re-render it there, so the episode's geometry is consistent "
             "against one reference (default: off). Requires a fixed camera, "
             "and is refused on a wrist camera by name. Off by default because "
             "watching the output was what settled it: the registration "
             "measures better on the background, but it resamples every frame, "
             "and the result reads as shakier than the frames as the model "
             "gave them.",
    )
    parser.add_argument(
        "--min-inliers", type=float, default=DEFAULT_MIN_INLIERS,
        help="Per frame: the fraction of sampled correspondences that must "
             "agree before a frame is aligned. A frame below it is packed "
             "unaligned and recorded as NaN. Calibrated on long-gap anchor "
             "registrations, which read far lower than adjacent-frame ones "
             "(default %(default)s).",
    )
    parser.add_argument(
        "--min-registered-frac", type=float, default=DEFAULT_MIN_REGISTERED_FRAC,
        help="Per episode: the fraction of sampled frames that must register "
             "before the camera is accepted as fixed. Lower it only with "
             "--min-inliers (default %(default)s).",
    )
    parser.add_argument(
        "--save-viz", action="store_true",
        help="Also write a side-by-side mp4 of each episode: the decoded "
             "input frames on the left, the packed output on the right, "
             "H.264 at --viz-crf. For looking at what the tool did; the .mkv "
             "beside it stays the lossless artifact.",
    )
    parser.add_argument(
        "--viz-crf", type=int, default=DEFAULT_VIZ_CRF,
        help=f"H.264 quality/size dial for --save-viz; higher is smaller and "
             f"worse (default {DEFAULT_VIZ_CRF}).",
    )
    parser.add_argument(
        "--range-sample-frames", type=int, default=DEFAULT_RANGE_SAMPLE_FRAMES,
        help=f"Frames inferred to fix each episode's depth range "
             f"(default {DEFAULT_RANGE_SAMPLE_FRAMES}).",
    )
    parser.add_argument(
        "--depth-min", type=float, default=DEFAULT_Z_MIN,
        help=f"Lower rail for the packed depth, metres (default {DEFAULT_Z_MIN}).",
    )
    parser.add_argument(
        "--depth-max", type=float, default=DEFAULT_Z_MAX,
        help=f"Upper rail for the packed depth, metres (default {DEFAULT_Z_MAX}).",
    )
    parser.add_argument(
        "--pt2", default=DEFAULT_PT2,
        help=f"Exported graph checkpoint (default {DEFAULT_PT2}).",
    )
    parser.add_argument(
        "--refiner", default=None,
        help="Refiner companion checkpoint (defaults to the .pt2 path with "
             "_refiner.pt).",
    )
    parser.add_argument(
        "--input-size", type=int, nargs=2, default=list(DEFAULT_INPUT_SIZE),
        metavar=("W", "H"),
        help=f"Resolution the checkpoint was exported for; must match it, and "
             f"sets the output video's size (default {DEFAULT_INPUT_SIZE[0]} "
             f"{DEFAULT_INPUT_SIZE[1]}).",
    )
    parser.add_argument("--device", default="cuda", help="Device (must be CUDA).")
    parser.add_argument("--refine_steps", type=int, default=1, help="Sparse refinement steps.")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="Encoder threads; default lets ffmpeg choose.",
    )
    args = parser.parse_args()

    # Refused by name, before anything is loaded.  Deciding it by measurement
    # would work on today's clips and quietly stop working on one where the arm
    # happens to hold still for the sampled frames -- and the failure it guards
    # against is invisible downstream, because the RANSAC returns a
    # plausible-looking transform for a camera that moved just as readily as
    # for one that did not.
    if args.align and WRIST_CAMERA_MARKER in args.camera.lower():
        parser.error(
            f"--camera {args.camera!r} is mounted on the arm, so it moves with "
            f"it and --align cannot work on it: alignment registers each frame "
            f"against the episode's first by matching the pixel grid, which "
            f"only means something for a camera that stayed put. Pass "
            f"--no-align to pack each frame as the model gave it."
        )

    if args.stride < 1:
        parser.error(f"--stride must be >= 1, got {args.stride}")
    if args.limit is not None and args.limit < 1:
        parser.error(f"--limit must be >= 1, got {args.limit}")
    if args.range_sample_frames < 1:
        parser.error(f"--range-sample-frames must be >= 1, got {args.range_sample_frames}")
    if not 0.0 <= args.min_registered_frac <= 1.0:
        parser.error(
            f"--min-registered-frac must be in [0, 1], got "
            f"{args.min_registered_frac}"
        )
    if not 0.0 <= args.min_inliers <= 1.0:
        parser.error(f"--min-inliers must be in [0, 1], got {args.min_inliers}")
    if args.depth_min <= 0.0 or args.depth_min >= args.depth_max:
        parser.error(
            f"--depth-min ({args.depth_min}) must be > 0 and < --depth-max "
            f"({args.depth_max}): the packed depth channel is log-encoded "
            f"over that range."
        )
    return args


def plan_output_paths(
    wrapper: LiberoWrapper, episode: Episode, out_dir: Path
) -> tuple[Path, Path]:
    """Where an episode-camera's ``.mkv`` and its ``.npz`` sidecar go.

    The dataset name and the camera key are kept in the path, so one output
    tree can hold every dataset and camera without them colliding --
    ``episode_000000`` exists in all four libero datasets, and both of them
    have a wrist camera.
    """
    mkv = Path(out_dir) / wrapper.name / wrapper.camera / (
        f"episode_{episode.index:06d}.mkv"
    )
    return mkv, mkv.with_suffix(".npz")


def plan_viz_path(mkv_path: Path) -> Path:
    """Where the side-by-side visualization mp4 for an episode goes.

    Beside the ``.mkv`` and named off its stem, so the pair stays together and
    a ``.npz``, a ``.mkv`` and a ``.viz.mp4`` can never collide.
    """
    return mkv_path.with_name(mkv_path.stem + ".viz.mp4")


def _fit_to(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """*rgb* resized to *size* ``(H, W)``, unchanged when it already fits.

    Only the visualization needs this: the dataset's frames and the model's
    output are the same 512x512 on the shipped data, but ``--input-size`` can
    separate them, and a side-by-side of two different heights is not a
    picture. Resized for looking at, so the interpolation is not load-bearing.
    """
    if rgb.shape[:2] == tuple(size):
        return rgb
    return cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def _dry_run(episodes: Sequence[Episode], wrapper: LiberoWrapper,
             args: argparse.Namespace) -> None:
    total = 0
    for episode in episodes:
        kept = len(range(episode.from_index, episode.to_index, args.stride))
        if args.max_frames:
            kept = min(kept, args.max_frames)
        total += kept

    width, height = args.input_size
    raw = total * width * height * 3
    print(
        f"\n{len(episodes)} episode(s) of {wrapper.name!r}, "
        f"{total} frame(s) at --stride {args.stride}\n"
        f"camera {wrapper.camera}, alignment "
        f"{'on' if args.align else 'off'}"
        + (f", visualization at --viz-crf {args.viz_crf}" if args.save_viz else "")
        + "\n"
        f"output size {width}x{height} (the checkpoint's --input-size)\n"
        f"raw pixel data {raw / 2**30:.1f} GiB — an upper bound. FFV1 "
        f"compresses this; the real total will be smaller."
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    wrapper = LiberoWrapper(args.repo_id, args.data_root, args.camera)
    episodes = wrapper.episodes(args.episode_idxes)
    if args.limit:
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit(f"No episodes selected in {wrapper.name!r}.")

    if args.dry_run:
        _dry_run(episodes, wrapper, args)
        return

    model = MoGev3_PT2(
        args.pt2,
        refiner_path=args.refiner,
        device=args.device,
        refine_steps=args.refine_steps,
        input_size=tuple(args.input_size),
    )
    pack = NormalDepthPack(pole=MOGE_POLE, z_min=args.depth_min, z_max=args.depth_max)

    written = skipped = failed = 0
    frames_total = 0
    started = time.time()
    for index, episode in enumerate(episodes, 1):
        mkv_path, npz_path = plan_output_paths(wrapper, episode, out_dir)
        if args.skip_existing and mkv_path.exists() and npz_path.exists():
            # A run that asked for the visualization is not finished without
            # it, or adding --save-viz to an existing output tree would skip
            # everything and quietly produce nothing.
            if not args.save_viz or plan_viz_path(mkv_path).exists():
                skipped += 1
                continue

        try:
            count, registered, considered = process_episode(
                model, pack, wrapper, episode, mkv_path, npz_path, args)
        except Exception as error:  # noqa: BLE001 -- one bad episode must not
            # end a multi-hour batch. The failure is named, counted, and the
            # run exits non-zero below, so it cannot pass unnoticed.
            failed += 1
            print(f"[{index}/{len(episodes)}] FAILED episode "
                  f"{episode.index}: {error}")
            continue

        written += 1
        frames_total += count
        elapsed = time.time() - started
        # The registration rate is printed even on success: an episode that
        # aligned on only just enough frames is the one a caller would
        # otherwise only find out about from the sidecar.
        align = (f", registered {registered}/{considered}" if considered else "")
        print(
            f"[{index}/{len(episodes)}] episode {episode.index} "
            f"({episode.length} frames, {episode.task[:48]!r}) — {count} "
            f"written{align}, {elapsed / written:.1f} s/episode, "
            f"{elapsed / 60:.1f} min elapsed"
        )

    print(
        f"\nwrote {written} episode(s) / {frames_total} frame(s) to {out_dir}"
        + (f", skipped {skipped} already present" if skipped else "")
        + (f", {failed} FAILED" if failed else "")
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
