"""Tests for the pure helpers behind tools/general_test/module/infer_moge3_video.py.

The model and ffmpeg paths need a GPU and real videos, so they are verified by
running the tool rather than here.  What is tested here is everything that can
be decided on the CPU: where an episode-camera's output lands, which frames an
episode contributes, which of them get sampled for the episode's depth range,
what that range is, and the geometry a frame is packed from.

The geometry section carries the load.  ``reproject_to_camera``'s projection is
the exact inverse of the unprojection, so an identity registration has to hand
back the maps it was given, bit for bit; that invariant is pinned here because
everything the aligned half of the tool writes inherits the convention it
encodes.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "general_test" / "module"))

from infer_moge3_video import (  # noqa: E402
    Anchor,
    FramePrediction,
    _kept_frames,
    encoded_scale,
    frame_geometry,
    plan_output_paths,
    resolve_video_range,
    sample_frame_indices,
)

from utils.libero_wrapper import Episode  # noqa: E402
from utils.normal_depth_pack import MOGE_POLE, DepthScale, NormalDepthPack  # noqa: E402
from utils.visualize.moge_register import Registration, reproject_to_camera  # noqa: E402

DATASET = "libero_goal_no_noops_lerobot"
OTHER_DATASET = "libero_object_no_noops_lerobot"
CAMERA = "observation.images.image"
WRIST_CAMERA = "observation.images.wrist_image"
FPS = 20.0


# ----------------------------------------------------------------- fixtures

def _episode(index=0, from_index=0, length=138, task="open the middle drawer"):
    """One ``Episode``, built directly rather than read from a dataset.

    It is a frozen dataclass of plain values -- the tools only ever read it --
    so episode handling can be driven with no dataset on disk.  ``to_index``
    follows from the length and the timestamps from the index, the way the
    wrapper's own table lays them out.
    """
    return Episode(
        index=index,
        from_index=from_index,
        to_index=from_index + length,
        length=length,
        from_timestamp=from_index / FPS,
        to_timestamp=(from_index + length) / FPS,
        task=task,
    )


def _wrapper(name=DATASET, camera=CAMERA):
    """A stand-in for the ``LiberoWrapper`` attributes the tools read off it.

    Constructing the real one needs a LeRobot dataset on disk, and the two
    attributes below are all ``plan_output_paths`` touches.
    """
    return SimpleNamespace(name=name, camera=camera)


def _args(stride=1, max_frames=None):
    """The two fields ``_kept_frames`` reads off the parsed arguments."""
    return SimpleNamespace(stride=stride, max_frames=max_frames)


# ------------------------------------------------------------------- output

def test_output_paths_keep_the_dataset_and_the_camera(tmp_path):
    """The name and the camera key survive into the path, and the episode
    becomes a zero-padded file name.
    """
    mkv, npz = plan_output_paths(_wrapper(), _episode(index=400), tmp_path)

    assert mkv == tmp_path / DATASET / CAMERA / "episode_000400.mkv"
    assert npz == mkv.with_suffix(".npz")


def test_output_paths_cannot_collide(tmp_path):
    """Every (dataset, camera, episode) triple gets its own pair of files.

    ``episode_000000`` exists in all four libero datasets and both of them have
    a wrist camera, so an episode index alone is nowhere near unique -- a run
    over two cameras that dropped either part of the path would overwrite its
    own output.
    """
    paths = set()
    for name in (DATASET, OTHER_DATASET):
        for camera in (CAMERA, WRIST_CAMERA):
            for index in (0, 1):
                paths.update(plan_output_paths(_wrapper(name, camera),
                                               _episode(index=index), tmp_path))

    assert len(paths) == 2 * 2 * 2 * 2


# ------------------------------------------------------------- kept frames

def test_kept_frames_are_absolute_and_stop_at_the_episode_end():
    """The frames come from the episode's own half-open index range.

    Absolute dataset indices, not episode-relative ones: the episode starts at
    138 in the dataset and the first frame kept is 138.  The stride walks off
    the end of the range rather than rounding up to it, so a strided episode
    never reaches into the next episode's frames.
    """
    episode = _episode(index=1, from_index=138, length=130)

    assert _kept_frames(episode, _args())[0] == 138
    assert len(_kept_frames(episode, _args())) == 130
    assert _kept_frames(episode, _args())[-1] == 267

    strided = _kept_frames(episode, _args(stride=4))
    assert strided == list(range(138, 268, 4))
    assert strided[-1] == 266  # 138 + 128, not the episode's last frame


def test_kept_frames_cap_applies_to_the_strided_frames():
    """``--max-frames`` counts frames that will be decoded.

    The cap lands after the stride, so ten frames at ``--stride 3`` is the
    first thirty of the episode rather than its first ten -- which is what a
    small first run wants, since the frames outside the selection are never
    read.
    """
    episode = _episode(from_index=0, length=100)

    assert _kept_frames(episode, _args(stride=3, max_frames=10)) == [
        0, 3, 6, 9, 12, 15, 18, 21, 24, 27
    ]


def test_kept_frames_refuses_an_empty_selection():
    """An episode with no frames in range is an error, not an empty video.

    Left alone it would reach the encoder as a zero-frame write, which reads
    like an ffmpeg problem rather than the selection mistake it is.
    """
    with pytest.raises(RuntimeError, match="no frames"):
        _kept_frames(_episode(length=0), _args())


# ------------------------------------------------------------------ sampling

def test_sample_spans_the_whole_episode_inclusively():
    idx = sample_frame_indices(138, 8)
    assert len(idx) == 8
    assert idx[0] == 0
    assert idx[-1] == 137
    assert idx == sorted(set(idx))


def test_sample_of_everything_when_k_meets_the_length():
    assert sample_frame_indices(5, 5) == [0, 1, 2, 3, 4]
    assert sample_frame_indices(5, 99) == [0, 1, 2, 3, 4]


def test_sample_never_repeats_an_index_on_a_short_episode():
    """The indices are distinct and in range for every shape of request.

    The caller infers each one, so a repeat would be a redundant inference and
    an out-of-range index a wasted decode.  Small n with large k takes the
    early return; n=7 with k=2 and k=5 exercises the linspace path.
    """
    for n in (1, 2, 3, 4, 7):
        for k in (2, 5, 8):
            idx = sample_frame_indices(n, k)
            assert idx == sorted(set(idx)), f"repeat at n={n} k={k}: {idx}"
            assert 0 <= idx[0] and idx[-1] < n, f"out of range at n={n} k={k}: {idx}"


def test_sample_of_an_empty_episode():
    assert sample_frame_indices(0, 8) == []


# --------------------------------------------------------------------- range

def _pack(**kw) -> NormalDepthPack:
    return NormalDepthPack(pole=MOGE_POLE, z_min=0.25, z_max=3.0, **kw)


def _frame(lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """A frame whose usable depths run from *lo* to *hi*."""
    depth_z = np.array([[lo, hi]], dtype=np.float32)
    return depth_z, np.ones_like(depth_z, dtype=bool)


def test_range_is_the_union_of_the_sampled_frames():
    pack = _pack()
    z_min, z_max = resolve_video_range(pack, [_frame(0.8, 2.0), _frame(0.5, 2.5)])
    assert (z_min, z_max) == pytest.approx((0.5, 2.5))


def test_range_never_escapes_the_rails():
    """The rails are a hard bound: the model's depth can run past them, and a
    range outside them would divide by a span the codec cannot encode."""
    pack = _pack()
    z_min, z_max = resolve_video_range(pack, [_frame(0.05, 9.0)])
    assert (z_min, z_max) == (0.25, 3.0)


def test_range_falls_back_to_the_rails_for_an_unusable_sample():
    pack = _pack()
    assert resolve_video_range(pack, []) == (0.25, 3.0)

    empty = (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool))
    assert resolve_video_range(pack, [empty]) == (0.25, 3.0)

    # A single usable sample beside an unusable one still decides the range --
    # the unusable frame contributes nothing rather than dragging in the rails.
    assert resolve_video_range(pack, [_frame(0.9, 1.4), empty]) == pytest.approx((0.9, 1.4))


def test_range_ignores_non_finite_and_non_positive_depth():
    """log of zero or a negative is undefined, so those pixels are not depths
    and must not pull the range down to the rail."""
    pack = _pack()
    depth_z = np.array([[0.0, -1.0], [np.nan, 1.2]], dtype=np.float32)
    other = np.full((2, 2), 1.7, dtype=np.float32)
    valid = np.ones((2, 2), dtype=bool)
    got = resolve_video_range(pack, [(depth_z, valid), (other, valid)])
    assert got == pytest.approx((1.2, 1.7))


def test_range_falls_back_when_the_sample_collapses_to_a_point():
    """A flat sample would divide by a zero span."""
    pack = _pack()
    assert resolve_video_range(pack, [_frame(1.5, 1.5)]) == (0.25, 3.0)


def test_one_frame_episode_range_matches_the_codec_exactly():
    """The property the whole sampling scheme rests on.

    With one sampled frame the episode's range must be the range the codec
    would have picked for that frame alone -- that is what makes the video tool
    and the per-image tool agree bit for bit on a single-frame input.
    """
    pack = _pack()
    rng = np.random.default_rng(0)
    depth_z = (rng.random((16, 16), dtype=np.float32) * 2.0 + 0.4).astype(np.float32)
    valid = np.ones_like(depth_z, dtype=bool)

    assert resolve_video_range(pack, [(depth_z, valid)]) == pack.resolve_range(depth_z, valid)


def test_range_honours_the_valid_mask():
    """The mask is the model's own mask; a pixel it excluded must not widen
    the range the episode is encoded over."""
    pack = _pack()
    depth_z = np.array([[0.8, 2.9], [1.0, 2.9]], dtype=np.float32)
    valid = np.array([[True, False], [True, False]])
    # Without the mask the far column would widen this to (0.8, 2.9).
    assert resolve_video_range(pack, [(depth_z, valid)]) == pytest.approx((0.8, 1.0))


# ----------------------------------------------------------- frame geometry

#: Normalized-uv intrinsics, the convention ``FramePrediction.intrinsics`` is
#: in: ``fx = k[0,0]``, ``cx = k[0,2] = 0.5``, pixel centres at
#: ``(j + 0.5) / W``.  Deliberately off 1.0 and asymmetric between the axes, so
#: a transposed fx/fy or a swapped row and column would move the pixels.
K = np.array([[0.9, 0.0, 0.5], [0.0, 1.1, 0.5], [0.0, 0.0, 1.0]])

#: A second camera, so that a test can tell which of the two intrinsics a
#: re-render was actually projected through.
OTHER_K = np.array([[1.3, 0.0, 0.5], [0.0, 0.7, 0.5], [0.0, 0.0, 1.0]])

H, W = 8, 12  # non-square, so a transposed axis shows up


def _prediction(height=H, width=W, seed=3, intrinsics=None, holes=False,
                shift=0.05, metric_scale=1.25) -> FramePrediction:
    """A synthetic model output, unprojected the way MoGe unprojects.

    ``points`` is built from the pixel grid through ``intrinsics`` --
    ``X = (u - cx) / fx * Z``, with the centre of pixel ``j`` at
    ``(j + 0.5) / W`` -- rather than from points chosen freely, so
    ``points[..., 2] == depth_m`` and every point projects back to its own
    pixel.  That is what makes an identity registration an exact round trip
    rather than a close one.

    The unprojection runs in float32, the dtype ``reproject_to_camera`` keeps
    the points in.  Doing it in double precision would hand the float32
    projection an already-rounded input and cost the round trip its exactness.
    """
    k = np.asarray(K if intrinsics is None else intrinsics, dtype=np.float64)
    fx, fy = np.float32(k[0, 0]), np.float32(k[1, 1])
    cx, cy = np.float32(k[0, 2]), np.float32(k[1, 2])

    j, i = np.meshgrid(np.arange(width), np.arange(height))
    u = ((j + 0.5) / width).astype(np.float32)
    v = ((i + 0.5) / height).astype(np.float32)
    depth = (0.6 + 2.4 * np.random.default_rng(seed).random((height, width))).astype(np.float32)
    points = np.stack(
        [(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=-1
    ).astype(np.float32)

    normals = np.random.default_rng(seed + 1).normal(size=(height, width, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    valid = np.ones((height, width), dtype=bool)
    if holes:
        valid[0, :] = False
        valid[3, 2] = False

    return FramePrediction(
        depth_m=np.where(valid, depth, 0.0).astype(np.float32),
        # A pixel with no geometry has no normal either: the renderer writes
        # 0.0 there and the packer zeroes the whole pixel, so leaving the
        # model's own normal in place would leave the fixture with something no
        # re-render could reproduce.
        normal=np.where(valid[..., None], normals, 0.0).astype(np.float32),
        valid=valid,
        points=np.where(valid[..., None], points, np.nan).astype(np.float32),
        intrinsics=k,
        shift=shift,
        metric_scale=metric_scale,
    )


def _rotation(axis, degrees) -> np.ndarray:
    """Rodrigues rotation matrix, for a registration that is not the identity."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degrees)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def test_anchor_scale_is_the_anchor_frame_s_own_fit():
    """The episode's single metric basis, read off the anchor's prediction.

    Taking it from the anchor rather than from the frame in hand is the whole
    reason a registered episode carries one scalar pair instead of a per-frame
    array: a per-frame pair is what would modulate the packed brightness frame
    to frame and flicker a static scene.
    """
    prediction = _prediction(shift=-0.25, metric_scale=1.4)
    anchor = Anchor(index=42, prediction=prediction)

    assert anchor.index == 42
    assert anchor.prediction is prediction

    scale = anchor.scale()
    assert isinstance(scale, DepthScale)
    assert (scale.shift, scale.metric_scale) == (-0.25, 1.4)
    assert scale != DepthScale()  # not the identity default an unaligned run keeps


def test_encoded_scale_is_the_anchor_s_basis_when_aligning():
    """One basis for the whole episode, so the packed channel cannot breathe.

    The frame's own fit is made deliberately different from the anchor's: if
    the encoder used it, every frame would be packed through a different affine
    and the depth channel's brightness would swing with `metric_scale`.
    """
    anchor = Anchor(index=0, prediction=_prediction(seed=1, shift=0.05,
                                                    metric_scale=1.25))
    frame = _prediction(seed=2, shift=0.90, metric_scale=0.40)

    chosen = encoded_scale(anchor, frame)

    assert chosen.shift == pytest.approx(0.05)
    assert chosen.metric_scale == pytest.approx(1.25)


def test_encoded_scale_is_the_frame_s_own_fit_without_an_anchor():
    """``--no-align`` keeps each frame in the model's own affine, as it always
    did — and *not* the identity.

    This is the survey's range bug pinned. The range has to be measured over
    the depths the encoder actually packs, which are relative; measuring it on
    the raw metric depth widened the range by each frame's ``metric_scale`` and
    spent codes on nothing. Both loops now go through this one function, so the
    identity here would put them back out of step.
    """
    frame = _prediction(seed=2, shift=0.90, metric_scale=0.40)

    chosen = encoded_scale(None, frame)

    assert chosen.shift == pytest.approx(0.90)
    assert chosen.metric_scale == pytest.approx(0.40)


def test_frame_geometry_without_a_registration_keeps_the_model_s_own_maps():
    """``--no-align``, and the fallback for a frame whose fit was refused.

    The anchor here is a different frame seen by a different camera.  It is
    what the aligned path re-renders into, and the unaligned path must not
    touch it: a frame re-packed in another frame's camera is exactly what
    ``--no-align`` says not to do.
    """
    prediction = _prediction()
    anchor = Anchor(index=1, prediction=_prediction(seed=11, intrinsics=OTHER_K))

    depth, normals, valid, holes = frame_geometry(
        prediction, anchor.prediction, None, (H, W)
    )

    assert np.array_equal(depth, prediction.depth_m)
    assert np.array_equal(normals, prediction.normal)
    assert np.array_equal(valid, prediction.valid)
    # NaN, not 0.0: nothing was re-rendered here, so there is no hole count to
    # report. A zero would be indistinguishable from a re-render that happened
    # to lose nothing, which is the opposite claim.
    assert np.isnan(holes)


def test_frame_geometry_identity_registration_reproduces_the_frame_bit_for_bit():
    """The oracle: the projection is the exact inverse of the unprojection.

    An identity similarity -- scale 1, identity rotation, no translation --
    with the anchor's intrinsics the frame's own must give the maps back
    unchanged.  This is asserted exactly rather than to a tolerance, because a
    tolerance would hide precisely what it exists to catch: a transposed
    fx/fy, a pixel centre half a pixel out, or a swapped row and column all
    leave a re-render that is merely *close*.
    """
    prediction = _prediction()
    anchor = Anchor(index=0, prediction=prediction)
    identity = Registration(scale=1.0, rotation=np.eye(3), translation=np.zeros(3),
                            inliers=1.0, n_points=int(prediction.valid.sum()))

    depth, normals, valid, holes = frame_geometry(
        prediction, anchor.prediction, identity, (H, W)
    )

    assert np.array_equal(depth, prediction.depth_m)
    assert np.array_equal(normals, prediction.normal)
    assert np.array_equal(valid, prediction.valid)
    assert holes == 0.0


def test_frame_geometry_keeps_the_model_s_holes_as_holes():
    """A pixel the model had nothing to say about comes back a hole.

    The re-render does not fill one -- there is no geometry to fill it with --
    and the count it reports is over the whole output grid, so the frame's own
    holes are in it.  That is what makes the sidecar's ``align_holes`` a
    measure of how much of the frame is missing rather than of how much the
    re-render lost.
    """
    prediction = _prediction(holes=True)
    anchor = Anchor(index=0, prediction=prediction)
    identity = Registration(scale=1.0, rotation=np.eye(3), translation=np.zeros(3),
                            inliers=1.0, n_points=1000)

    depth, normals, valid, holes = frame_geometry(
        prediction, anchor.prediction, identity, (H, W)
    )

    assert not prediction.valid.all(), "the fixture must have holes to test with"
    assert np.array_equal(valid, prediction.valid)
    assert np.array_equal(depth[valid], prediction.depth_m[valid])
    assert np.array_equal(normals[valid], prediction.normal[valid])
    assert np.array_equal(depth[~valid], np.zeros_like(depth[~valid]))
    assert np.array_equal(normals[~valid], np.zeros_like(normals[~valid]))
    assert holes == pytest.approx(1.0 - prediction.valid.mean())


def test_frame_geometry_applies_the_registration_to_the_maps():
    """A registration that is not the identity moves the frame, not just its
    shape: a fit at twice the anchor's scale doubles every depth.

    The scale is dyadic, so the comparison stays exact.  Breaks if the
    registration is dropped on the way to the renderer and the frame is packed
    as it came out of the model -- which on an aligned episode is the
    frame-to-frame breathing the alignment exists to remove.
    """
    prediction = _prediction()
    anchor = Anchor(index=0, prediction=prediction)
    scaled = Registration(scale=2.0, rotation=np.eye(3), translation=np.zeros(3),
                          inliers=1.0, n_points=1000)

    depth, _, valid, holes = frame_geometry(prediction, anchor.prediction, scaled, (H, W))

    assert np.array_equal(depth, 2.0 * prediction.depth_m)
    assert np.array_equal(valid, prediction.valid)
    assert holes == 0.0


def test_frame_geometry_hands_the_renderer_the_registration_and_the_anchor_camera():
    """Delegation, pinned argument by argument.

    The properties above are what matters; this is the test that catches a call
    wired the wrong way round -- a swapped scale and translation, or the two
    cameras' intrinsics in each other's slots.  Both return four arrays of the
    right shape and look entirely reasonable.

    The two intrinsics differ here on purpose, because the slots are not
    interchangeable: the anchor's is the camera being rendered *into*, and the
    frame's is the one its points were unprojected through -- which MoGe
    refits every frame, so it is not the anchor's to substitute.
    """
    prediction = _prediction()
    anchor = Anchor(index=3, prediction=_prediction(seed=11, intrinsics=OTHER_K))
    registration = Registration(
        scale=1.5,
        rotation=_rotation([0.2, 0.5, -0.8], 3.0),
        translation=np.array([0.01, -0.02, 0.03]),
        inliers=0.9,
        n_points=1234,
    )

    got = frame_geometry(prediction, anchor.prediction, registration, (H, W))
    want = reproject_to_camera(
        prediction.points,
        prediction.normal,
        prediction.valid,
        registration.scale,
        registration.rotation,
        registration.translation,
        anchor.prediction.intrinsics,
        (H, W),
        source_intrinsics=prediction.intrinsics,
    )

    for got_part, want_part in zip(got[:3], want[:3]):
        assert np.array_equal(got_part, want_part)
    assert got[3] == want[3]
