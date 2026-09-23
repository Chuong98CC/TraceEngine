"""The packed normal + depth codec (``utils.normal_depth_pack``).

Pure numpy, so these run on CPU.
"""

import cv2
import numpy as np
import pytest

from utils.normal_depth_pack import (
    DAB_DETAIL_D,
    DAB_DETAIL_SIGMA_COLOR,
    DAB_DETAIL_SIGMA_SPACE,
    DAB_L_FAR,
    DAB_L_NEAR,
    DAB_L_PER_CODE,
    DEPTH_CODE_MAX,
    DEPTH_CODE_MIN,
    DEPTH_LEVELS,
    MOGE_POLE,
    DepthScale,
    NormalDepthPack,
    decode_alb_normal,
    decode_dab,
    decode_l_normal,
    encode_alb_normal,
    encode_dab,
    encode_fused,
    encode_l_normal,
    estimate_pole,
    pole_conditioning,
)

#: Rails the depth channel is encoded over, in model depth units.
Z_MIN, Z_MAX = 0.25, 3.0

#: Codec quantization: 254 levels span log(3.0) - log(0.25) = 2.4849 nats, so
#: one level is 0.009783 nats -> 0.983% relative in depth. Hand-derived from
#: the range, not read off the codec, with margin for the floor.
DEPTH_RTOL = 0.012

#: A mid-range depth, used whenever the depth channel is not the subject.
MID_Z = 1.0

#: The Dab carrier's own tolerance. Twice the codec's, because its ramp is
#: compressed into L* 20-85: each code is a smaller step in L and the 8-bit
#: round trip resolves it less reliably, which costs a second code at the ends.
#: That is the price of the ramp not running from near-black to near-white.
DAB_RTOL = 2 * DEPTH_RTOL

MOGE = NormalDepthPack(pole=MOGE_POLE, z_min=Z_MIN, z_max=Z_MAX)

#: Same codec with the pole guard off. Most of these tests drive the codec with
#: adversarial fixtures — axis normals spanning the whole sphere — which sit
#: genuinely far from the pole and would warn every time. The guard has its own
#: tests below; everywhere else it is noise.
CODEC = NormalDepthPack(
    pole=MOGE_POLE, z_min=Z_MIN, z_max=Z_MAX, check_conditioning=False
)


def _unit(normals) -> np.ndarray:
    """Normalize last-axis vectors (hand-built fixtures -> unit normals)."""
    normals = np.asarray(normals, dtype=np.float32)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _encode(normals, depth_z=None, valid=None, pack=CODEC) -> np.ndarray:
    """Encode with a mid-range depth unless one is given.

    The range is pinned rather than adaptive: these tests are about the codec,
    so the encoded span must not depend on the fixture's own extent.
    """
    normals = np.asarray(normals, dtype=np.float32)
    if depth_z is None:
        depth_z = np.full(normals.shape[:2], MID_Z, dtype=np.float32)
    return pack.encode(normals, np.asarray(depth_z), valid, z_range=(Z_MIN, Z_MAX))


def _decode(image, pack=CODEC):
    return pack.decode(image, Z_MIN, Z_MAX)


def test_encode_shape_and_dtype():
    """The packed image is an (H, W, 3) uint8 RGB."""
    packed = _encode(_unit([[0, 0, -1], [1, 0, 0]]).reshape(1, 2, 3))

    assert packed.shape == (1, 2, 3)
    assert packed.dtype == np.uint8


@pytest.mark.parametrize(
    "pole",
    [(0, 0, -1), (0, 0, 1), (0.2, -0.5, -0.84), (1, 0, 0)],
    ids=["opencv", "opengl", "off-axis", "x-axis"],
)
def test_encode_puts_the_models_pole_normal_at_the_centre(pole):
    """Whatever direction a model calls camera-facing lands on the diamond centre.

    The octahedral map is well conditioned at its pole and degenerate at the
    opposite one, so the pole has to sit on the data. This is the whole
    generalization: declare the convention, and the codec puts the bulk of the
    image where the projection is locally near-identity.

    Breaks if the rotation carrying ``pole`` onto the projection axis is
    dropped, inverted, or built from the wrong direction.
    """
    pack = NormalDepthPack(pole=np.array(pole, dtype=np.float64),
                           z_min=Z_MIN, z_max=Z_MAX)
    normals = _unit(np.array([[pole]], dtype=np.float32))

    packed = _encode(normals, pack=pack)

    assert packed[0, 0, :2].tolist() == [127, 127]


def test_encode_axis_bytes_match_hand_derivation():
    """Axis normals land on their hand-derived octahedral cells.

    Derived by hand for a pole of (0, 0, -1), whose rotation is a 180 degree
    turn about x: normals along +x pass through untouched, while +y and -y
    swap. Breaks if the L1 normalization, the [-1, 1] -> [0, 255] rescale, the
    channel order (U, V, depth) or the pole rotation changes.
    """
    normals = np.array(
        [[[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]],
        dtype=np.float32,
    )

    packed = _encode(normals)

    # u = (px + 1) / 2 * 255, truncated to uint8. The centre is the
    # camera-facing normal, the corner is the one pointing away.
    assert packed[0, 0, :2].tolist() == [127, 127]  # -z (camera-facing) -> centre
    assert packed[0, 1, :2].tolist() == [255, 127]  # +x -> u max
    assert packed[0, 2, :2].tolist() == [127, 0]  # +y -> v min (the turn about x)
    assert packed[0, 3, :2].tolist() == [127, 255]  # -y -> v max
    assert packed[0, 4, :2].tolist() == [255, 255]  # +z -> far corner


def test_encode_keeps_a_flat_patch_tightly_clustered():
    """Normals within half a degree of each other must pack close together.

    ``(0, 0, -1)`` carries the bulk of a camera-frame model's output — every
    fronto-parallel surface lands on it. Put the projection pole anywhere else
    and that direction becomes the octahedral's four-way degenerate corner,
    where a hair of +nx versus -nx sends two neighbouring pixels of one flat
    surface to opposite corners of the square: measured at a 359-code jump
    between pixels whose normals differ by 0.56 degrees. In the rendered image
    that is a hard-edged blotch on every flat wall.

    Breaks if the pole moves off the data, however the pole is spelled.
    """
    tilt = np.radians(0.5)
    azimuths = np.radians(np.arange(0.0, 360.0, 45.0))
    normals = np.stack(
        [
            np.sin(tilt) * np.cos(azimuths),
            np.sin(tilt) * np.sin(azimuths),
            np.full_like(azimuths, -np.cos(tilt)),
        ],
        axis=-1,
    ).astype(np.float32)

    codes = _encode(normals.reshape(1, -1, 3))[0, :, :2].astype(np.float32)

    spread = float(np.linalg.norm(codes[:, None, :] - codes[None, :, :], axis=-1).max())
    assert spread < 20, (
        f"a 0.5 degree patch spread {spread:.0f} codes — the patch is sitting "
        f"on a degenerate point of the projection"
    )


def test_decode_hand_built_bytes():
    """Hand-built octahedral bytes decode to their hand-derived normals.

    Built from literals rather than by round-tripping through ``encode``, so it
    is the decode side's own channel order and rescale under test. Breaks if
    the U/V channels are read in the wrong order — swapping them is invisible
    to a round-trip, which just applies the same swap twice.
    """
    # u = 255 -> +1, v = 127 -> ~0, so the first pixel is the +x normal. The
    # depth channel is a valid mid-range code, not the invalid 0. The pole
    # rotation (a 180 degree turn about x) then sends +y to -y, so the middle
    # pixel comes back negated.
    packed = np.array(
        [[[255, 127, 128], [127, 255, 128], [127, 127, 128]]], dtype=np.uint8
    )

    normals, _, valid = _decode(packed)

    assert np.allclose(normals[0, 0], [1, 0, 0], atol=0.05)  # u max -> +x
    assert np.allclose(normals[0, 1], [0, -1, 0], atol=0.05)  # v max -> -y
    assert np.allclose(normals[0, 2], [0, 0, -1], atol=0.05)  # centre -> -z
    assert valid.all()


def test_round_trips_back_facing_normals():
    """Normals pointing away from the camera survive encode -> decode.

    These are the ones the projection has to fold: the pole sits on the
    camera-facing direction, so a normal pointing away from the camera is on
    the far side of it.

    Breaks if the folding scales by ``np.sign`` of the raw coordinate: sign(0)
    is 0, so a coordinate that is exactly 0 zeroes the other folded coordinate
    and the normal comes back rotated by up to 90 degrees.
    """
    normals = _unit([[[0, 0, 1], [0.001, 0, 0.999], [0, 0.001, 0.999]]])

    decoded, _, _ = _decode(_encode(normals))

    assert np.allclose(decoded, normals, atol=0.05)


def test_round_trips_random_unit_normals():
    """Both hemispheres survive encode -> decode within 8-bit octahedral error."""
    rng = np.random.default_rng(0)
    normals = _unit(rng.normal(size=(32, 48, 3)))

    decoded, _, _ = _decode(_encode(normals))

    assert np.allclose(decoded, normals, atol=0.05)


def test_invalid_pixels_are_zero():
    """Pixels outside the mask pack to (0, 0, 0) in every channel."""
    normals = np.zeros((1, 2, 3), dtype=np.float32)
    normals[0, 0] = [0, 0, -1]
    valid = np.array([[True, False]])

    packed = _encode(normals, np.full((1, 2), MID_Z, np.float32), valid)

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_masks_pixels_whose_value_alone_looks_valid():
    """A ``valid=False`` pixel packs to (0, 0, 0) even with an in-range depth.

    Breaks if the ``valid`` argument is ignored. The depth here is finite and
    mid-range on both pixels, so the codec's own depth guard cannot catch it —
    the mask is the only thing that can.
    """
    normals = np.array([[[0, 0, -1], [0, 0, -1]]], dtype=np.float32)

    packed = _encode(normals, np.full((1, 2), MID_Z, np.float32),
                     np.array([[True, False]]))

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "bad", [0.0, -1.0, np.nan, np.inf],
    ids=["zero", "negative", "nan", "inf"],
)
def test_unrepresentable_depths_are_invalid(bad):
    """Non-positive and non-finite depths pack to (0, 0, 0) with no mask given.

    ``log`` of any of these is undefined, so left alone they would clip to a
    rail and read back as a plausible surface at the wrong depth.

    Breaks if the validity test drops the ``> 0`` half or the finiteness half.
    """
    normals = np.array([[[0, 0, -1], [0, 0, -1]]], dtype=np.float32)

    packed = _encode(normals, np.array([[MID_Z, bad]], dtype=np.float32))

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_reserves_code_zero_for_invalid():
    """A valid pixel sitting exactly on ``z_min`` encodes to code 1, not 0.

    Code 0 is the invalid sentinel, so mapping the data onto [0, 255] would
    make every frame's *nearest* surfaces decode as masked-out. Breaks if the
    encoding loses the +1 offset.
    """
    normals = np.array([[[0, 0, -1]]], dtype=np.float32)

    packed = _encode(normals, np.array([[Z_MIN]], dtype=np.float32))

    assert packed[0, 0, 2] == 1
    _, _, valid = _decode(packed)
    assert valid.all(), "a valid pixel at z_min must not decode as invalid"


def test_clips_out_of_range_to_the_range_ends():
    """Valid values beyond the range clip to codes 1 / 255 and stay valid.

    A rail tighter than the scene must degrade a pixel to the range end, never
    to the invalid sentinel.
    """
    normals = np.array([[[0, 0, -1], [0, 0, -1]]], dtype=np.float32)

    packed = _encode(normals, np.array([[Z_MIN / 4, Z_MAX * 4]], dtype=np.float32))
    _, decoded, valid = _decode(packed)

    assert packed[0, 0, 2] == 1
    assert packed[0, 1, 2] == 255
    assert valid.all()
    assert np.allclose(decoded[0], [Z_MIN, Z_MAX], rtol=DEPTH_RTOL)


def test_round_trips_depth_z():
    """The depth channel round-trips within the codec's bound."""
    z = np.array([[0.3, 0.5, 1.0, 2.5]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, -1], dtype=np.float32), (1, 4, 1))

    _, decoded, valid = _decode(_encode(normals, z))

    assert valid.all()
    assert np.allclose(decoded, z, rtol=DEPTH_RTOL)


def test_channel_is_log_not_linear():
    """The geometric mid-range sits mid-scale: the channel is logarithmic.

    Breaks if the depth channel is swapped for a plain linear (z - min) /
    (max - min) normalization, which would put this depth at code 58.
    """
    geometric_mid = np.sqrt(Z_MIN * Z_MAX)  # 0.866
    normals = np.array([[[0, 0, -1]]], dtype=np.float32)

    packed = _encode(normals, np.array([[geometric_mid]], dtype=np.float32))

    # norm = 0.5 exactly -> 1 + 0.5 * 254 = 128.
    assert 127 <= packed[0, 0, 2] <= 129


def test_decode_pins_the_range_ends():
    """Codes 1 and 255 decode to exactly z_min and z_max.

    Pins the decode scaling, which the round-trip cannot: decoding over 255
    levels instead of 254 mis-scales by ~0.4%, and at the far end that is
    ~0.9% — inside the round-trip's tolerance. Breaks if the level count on
    either side of the packed image changes.
    """
    packed = np.array([[[127, 127, 1], [127, 127, 255]]], dtype=np.uint8)

    _, depth_z, valid = _decode(packed)

    assert valid.all()
    assert np.allclose(depth_z[0], [Z_MIN, Z_MAX], rtol=1e-6)


def test_decode_reconstructs_metric_depth():
    """packed + shift + metric_scale rebuilds the metric depth.

    This is the whole point of storing log depth: the sidecar's two scalars
    turn the packed channel back into metres. Uses real per-frame values.
    """
    shift, metric_scale = -0.1382, 0.8784
    z = np.array([[0.689, 1.500, 2.753]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, -1], dtype=np.float32), (1, 3, 1))

    _, decoded, valid = _decode(_encode(normals, z))
    metric = (decoded + shift) * metric_scale

    assert valid.all()
    assert np.allclose(metric, (z + shift) * metric_scale, rtol=DEPTH_RTOL)


def test_decode_marks_code_zero_invalid():
    """Code 0 unpacks as valid=False, everything else as True.

    Breaks if the validity test is loosened to ``code >= 0``, which on a uint8
    is always true — every masked-out pixel would then read as a real surface
    at 1 unit.
    """
    packed = np.array([[[127, 127, 0], [127, 127, 1]]], dtype=np.uint8)

    _, _, valid = _decode(packed)

    assert valid.tolist() == [[False, True]]


def test_resolve_range_tightens_to_the_data():
    """The frame's own depth range narrows the rails."""
    depth_z = np.array([[1.0, 2.0]], dtype=np.float32)

    assert np.allclose(MOGE.resolve_range(depth_z), [1.0, 2.0], rtol=1e-6)


def test_resolve_range_keeps_the_rails_when_they_are_tighter():
    """Rails inside the data range win — they are a hard limit."""
    depth_z = np.array([[0.1, 10.0]], dtype=np.float32)

    assert MOGE.resolve_range(depth_z) == (Z_MIN, Z_MAX)


def test_resolve_range_keeps_the_rails_when_disjoint():
    """Data entirely outside the rails falls back rather than inverting.

    Without the guard ``max(min) > min(max)`` and the range would be empty.
    """
    depth_z = np.array([[5.0, 6.0]], dtype=np.float32)

    assert MOGE.resolve_range(depth_z) == (Z_MIN, Z_MAX)


def test_resolve_range_keeps_the_rails_without_valid_pixels():
    """No valid pixels means nothing to tighten with."""
    depth_z = np.array([[1.0, 2.0]], dtype=np.float32)
    valid = np.array([[False, False]])

    assert MOGE.resolve_range(depth_z, valid) == (Z_MIN, Z_MAX)


def test_resolve_range_ignores_non_finite_pixels():
    """An infinite or NaN depth must not drag the range.

    Breaks if the finiteness guard is dropped: an inf would pin the maximum.
    """
    depth_z = np.array([[1.0, np.inf, np.nan, 2.0]], dtype=np.float32)

    lo, hi = MOGE.resolve_range(depth_z)

    assert np.allclose([lo, hi], [1.0, 2.0], rtol=1e-6)


def test_resolve_range_ignores_pixels_marked_invalid():
    """A finite value at an invalid pixel must not widen the range either.

    Breaks if the ``valid`` argument is dropped from the intersection.
    """
    depth_z = np.array([[1.0, 2.0, 2.5]], dtype=np.float32)
    valid = np.array([[True, False, True]])

    lo, hi = MOGE.resolve_range(depth_z, valid)

    assert np.allclose([lo, hi], [1.0, 2.5], rtol=1e-6)


def test_fixed_range_does_not_adapt():
    """``adaptive=False`` encodes over the rails regardless of the frame."""
    pack = NormalDepthPack(pole=MOGE_POLE, z_min=Z_MIN, z_max=Z_MAX, adaptive=False)
    depth_z = np.array([[1.0, 2.0]], dtype=np.float32)

    assert pack.resolve_range(depth_z) == (Z_MIN, Z_MAX)


# ------------------------------------------------------------- depth scale


def test_depth_scale_metric_is_the_identity():
    """A model already predicting metres needs shift 0 / metric_scale 1.

    This is why there is no metric-vs-relative mode flag: the affine form
    reduces to the identity, so one parameterization covers both.
    """
    scale = DepthScale()
    depth_z = np.array([0.5, 1.5, 3.0], dtype=np.float32)

    assert np.allclose(scale.to_metric(depth_z), depth_z)


def test_depth_scale_grounds_a_relative_model():
    """The affine case: metres = (depth_z + shift) * metric_scale, and back."""
    scale = DepthScale(shift=-0.1382, metric_scale=0.8784)
    depth_z = np.array([0.689, 1.500, 2.753], dtype=np.float32)

    metres = scale.to_metric(depth_z)

    # hand-derived: (0.689 - 0.1382) * 0.8784 = 0.5508 * 0.8784
    assert np.allclose(metres[0], 0.48382272, rtol=1e-6)
    assert np.allclose(scale.from_metric(metres), depth_z, rtol=1e-6)


def test_depth_scale_survives_the_sidecar(tmp_path):
    """A scale written to a sidecar loads back equal."""
    scale = DepthScale(shift=-0.1382, metric_scale=0.8784)
    path = tmp_path / "x_scale.npz"
    np.savez(path, **MOGE.scale_dict(scale, 0.689, 2.753, (4, 6)))

    back = DepthScale.from_dict(NormalDepthPack.load_scale(path))

    assert back == scale


# ------------------------------------------------------------ pole helpers


def _cluster(direction, count=4096, spread_deg=30.0, seed=0) -> np.ndarray:
    """A patch of unit normals clustered around ``direction``."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    rng = np.random.default_rng(seed)
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t1 = np.cross(d, ref)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(d, t1)

    tilt = np.radians(spread_deg) * np.sqrt(rng.random(count))
    azimuth = rng.random(count) * 2.0 * np.pi
    normals = (
        np.cos(tilt)[:, None] * d
        + np.sin(tilt)[:, None]
        * (np.cos(azimuth)[:, None] * t1 + np.sin(azimuth)[:, None] * t2)
    )
    return (normals / np.linalg.norm(normals, axis=-1, keepdims=True)).astype(np.float32)


@pytest.mark.parametrize(
    "direction",
    [(0, 0, -1), (0, 0, 1), (0.3, 0.4, -0.87)],
    ids=["opencv", "opengl", "off-axis"],
)
def test_estimate_pole_recovers_the_convention(direction):
    """The mean direction of a model's normals is its pole.

    This is how a new model's convention gets measured: run a few sample frames
    through :func:`estimate_pole`, then hard-code the answer.
    """
    target = np.asarray(direction, dtype=np.float64)
    target = target / np.linalg.norm(target)

    pole = estimate_pole(_cluster(target, spread_deg=10.0))

    assert np.allclose(pole, target, atol=0.02)


def test_estimate_pole_ignores_invalid_pixels():
    """Masked-out pixels must not drag the estimate.

    Breaks if ``valid`` is dropped: the bogus half here points the other way
    and would cancel the real signal.
    """
    good = _cluster((0, 0, -1), count=2048, spread_deg=5.0)
    bad = np.tile(np.array([0, 0, 1], dtype=np.float32), (2048, 1))
    normals = np.concatenate([good, bad]).reshape(2, -1, 3)
    valid = np.zeros((2, 2048), dtype=bool)
    valid[0] = True

    pole = estimate_pole(normals, valid)

    assert np.allclose(pole, [0, 0, -1], atol=0.02)


def test_estimate_pole_is_normalized_for_a_wide_spread():
    """The estimate comes back as a unit vector even for widely spread normals.

    Breaks if the mean is returned raw. A tight cluster has a mean of norm ~1
    already, so comparing it to the true direction cannot tell the two apart —
    but a hemisphere-wide spread has a mean of norm ~0.5, and that is off by
    half.
    """
    pole = estimate_pole(_cluster((0, 0, -1), spread_deg=90.0))

    assert np.isclose(np.linalg.norm(pole), 1.0, atol=1e-6)


def test_pole_conditioning_passes_a_well_placed_pole():
    """Data clustered on the projection pole is the good case."""
    report = pole_conditioning(_cluster((0, 0, -1)), pole=(0, 0, -1))

    assert report.ok
    assert report.centre_angle_deg < 20.0


def test_pole_conditioning_flags_a_pole_on_the_far_side():
    """The same normals against a pole 180 degrees away must be flagged.

    This is the regression guard for the bug this codec was built around:
    data on the far pole is *accurate* but blocky, so nothing about the decoded
    numbers reveals it. Only the amplification does.
    """
    normals = _cluster((0, 0, -1), spread_deg=10.0)

    good = pole_conditioning(normals, pole=(0, 0, -1))
    bad = pole_conditioning(normals, pole=(0, 0, 1))

    assert bad.ok is False
    assert bad.amplification > 3.0 * good.amplification, (
        f"far pole measured {bad.amplification:.2f} codes/deg against "
        f"{good.amplification:.2f} on the near pole"
    )


def test_encode_warns_when_the_pole_is_misplaced():
    """Encoding with a pole on the far side of the data warns.

    Breaks if the guard is dropped, which would leave a new model integration
    with accurate-but-blocky output and no signal about why.
    """
    pack = NormalDepthPack(pole=np.array([0.0, 0.0, 1.0]),
                           z_min=Z_MIN, z_max=Z_MAX)
    normals = _cluster((0, 0, -1)).reshape(1, -1, 3)
    depth_z = np.full((1, normals.shape[1]), MID_Z, dtype=np.float32)

    with pytest.warns(UserWarning, match="pole"):
        pack.encode(normals, depth_z)


def test_encode_is_silent_when_the_pole_is_right():
    """A well-placed pole encodes without warning."""
    import warnings

    normals = _cluster((0, 0, -1)).reshape(1, -1, 3)
    depth_z = np.full((1, normals.shape[1]), MID_Z, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MOGE.encode(normals, depth_z)


# ---------------------------------------------------------------- sidecar


def test_scale_dict_carries_what_decoding_needs():
    """The sidecar holds the range, the scale, the shape, the pole and extras.

    Without the pole a stale image decodes silently wrong, so it is part of the
    format rather than of the reader's configuration.
    """
    scale = DepthScale(shift=-0.1382, metric_scale=0.8784)

    d = MOGE.scale_dict(scale, 0.689, 2.753, (4, 6), intrinsics=np.eye(3))

    assert float(d["shift"]) == pytest.approx(-0.1382)
    assert float(d["metric_scale"]) == pytest.approx(0.8784)
    assert float(d["z_min"]) == pytest.approx(0.689)
    assert float(d["z_max"]) == pytest.approx(2.753)
    assert int(d["image_h"]) == 4
    assert int(d["image_w"]) == 6
    assert np.allclose(d["pole"], MOGE_POLE)
    assert np.allclose(d["intrinsics"], np.eye(3))


def test_decode_with_scale_reads_a_foreign_image(tmp_path):
    """An image decodes through another instance's sidecar, pole and all.

    The writer and reader here have deliberately different poles, so this
    passes only if ``decode_with_scale`` honours the *recorded* pole rather
    than the reading instance's. That is what makes the format self-describing.
    """
    writer = NormalDepthPack(pole=np.array([0.0, 0.0, -1.0]),
                             z_min=Z_MIN, z_max=Z_MAX)
    reader = NormalDepthPack(pole=np.array([0.0, 0.0, 1.0]),
                             z_min=Z_MIN, z_max=Z_MAX)
    normals = _cluster((0, 0, -1)).reshape(1, -1, 3)
    depth_z = np.full((1, normals.shape[1]), MID_Z, dtype=np.float32)

    image = writer.encode(normals, depth_z)
    path = tmp_path / "x_scale.npz"
    np.savez(path, **writer.scale_dict(DepthScale(), Z_MIN, Z_MAX, (1, normals.shape[1])))

    decoded, decoded_z, valid = reader.decode_with_scale(
        image, NormalDepthPack.load_scale(path)
    )

    assert valid.all()
    assert np.allclose(decoded, normals, atol=0.05)
    assert np.allclose(decoded_z, depth_z, rtol=DEPTH_RTOL)


# --------------------------------------------------------------------- Dab
# The depth-only carrier: the same depth code, ridden in the L channel of Lab
# while the input photo's own chroma supplies a and b (hence the name).  Where
# the octahedral image is a data container, this one is meant to still read as
# a picture, so the question these tests answer is how much depth accuracy the
# sRGB round trip costs -- and the answer has to be "one code", because the
# L channel is a full continuous channel and there is no reason to lose more.

#: One code of the log range, as a relative error in depth -- the same 0.983%
#: DEPTH_RTOL is built from.
ONE_CODE = np.exp(np.log(Z_MAX / Z_MIN) / (DEPTH_CODE_MAX - DEPTH_CODE_MIN)) - 1.0


def _dab_rgb(height: int = 48, width: int = 48) -> np.ndarray:
    """Colour fixture spanning both interesting cases.

    Most pixels are a smooth gradient of the sort a real frame is made of;
    the top row carries the fully saturated primaries, which are the pixels
    whose own chroma leaves the narrowest band of L representable.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            xs / (width - 1) * 255.0,
            ys / (height - 1) * 255.0,
            (1.0 - xs / (width - 1)) * 255.0,
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb[0, 0] = (255, 0, 0)
    rgb[0, 1] = (0, 255, 0)
    rgb[0, 2] = (0, 0, 255)
    rgb[0, 3] = (255, 255, 0)
    rgb[0, 4] = (0, 255, 255)
    rgb[0, 5] = (255, 0, 255)
    rgb[1, :, :] = np.linspace(0, 255, width).astype(np.uint8)[:, None]
    return rgb


def _dab_depth(height: int = 48, width: int = 48) -> np.ndarray:
    """Depths log-spaced across exactly [Z_MIN, Z_MAX].

    Spanning the range rather than sitting mid-range is the point: it puts
    pixels on both rails, so the code-1 end case is exercised by the core
    accuracy test instead of only by its own.
    """
    xs = np.linspace(np.log(Z_MIN), np.log(Z_MAX), width)
    return np.repeat(np.exp(xs)[None, :], height, axis=0).astype(np.float32)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB -> cv2 float Lab (L 0-100, a/b signed)."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def test_encode_dab_returns_uint8_rgb():
    """The blended image is an (H, W, 3) uint8 RGB, like every other output."""
    blended = encode_dab(_dab_rgb(4, 6), _dab_depth(4, 6), z_range=(Z_MIN, Z_MAX))

    assert blended.shape == (4, 6, 3)
    assert blended.dtype == np.uint8


def test_dab_round_trip_costs_at_most_two_codes_over_the_packed_image():
    """The carrier adds at most one code to what the packed image already costs.

    Measured against the packed image rather than against the original depth,
    because the depth channel's own encode *truncates*: comparing to the source
    would fold the codec's quantization floor in and let a two-code round trip
    pass. What is under test here is the Lab carrier alone.

    Breaks if the encoder writes a target L the pixel's own chroma cannot
    represent and lets the clip move it back: saturated pixels then lose up to
    a dozen codes (measured 12 on a real frame, against 1 for every frame once
    the chroma is reduced to fit).
    """
    rgb, depth_z = _dab_rgb(), _dab_depth()
    front_facing = np.broadcast_to([0.0, 0.0, -1.0], (*depth_z.shape, 3))
    _, packed_z, _ = CODEC.decode(
        CODEC.encode(front_facing, depth_z, z_range=(Z_MIN, Z_MAX)), Z_MIN, Z_MAX
    )

    # detail_alpha=0: this is the carrier's own cost, with the optional detail
    # layer off. Turning it on deliberately spends depth accuracy for surface
    # markings, which test_dab_*_detail_layer below covers.
    recovered, valid = decode_dab(
        encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0),
        Z_MIN,
        Z_MAX,
    )

    assert valid.all()
    rel = np.abs(recovered - packed_z) / packed_z
    assert rel.max() <= DAB_RTOL


def test_encode_dab_keeps_chroma_where_the_target_l_is_representable():
    """A pixel whose target L it can already represent comes back unchanged.

    Breaks if the encoder desaturates unconditionally -- scaling every pixel's
    chroma by some fixed factor -- rather than only where the gamut demands it.
    """
    rgb = np.full((2, 2, 3), (150, 120, 130), dtype=np.uint8)
    # Ask for exactly the L this colour already has, so the encoded pixel is
    # the original pixel and any change is the encoder's doing.
    own_l = float(_lab(rgb)[0, 0, 0])
    code = int(_ramp_code(np.float32(own_l)))
    depth_z = np.full(
        (2, 2),
        np.exp(np.log(Z_MIN) + (code - DEPTH_CODE_MIN) / DEPTH_LEVELS
               * np.log(Z_MAX / Z_MIN)),
        dtype=np.float32,
    )

    blended = encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX))

    # Asserted on a and b rather than on the bytes: snapping a continuous L onto
    # an integer code necessarily shifts the colour a unit or two, so a byte
    # comparison would be testing the L rounding, not the chroma.
    assert np.abs(_lab(blended)[0, 0, 1:] - _lab(rgb)[0, 0, 1:]).max() <= 2.0
    # The L that comes back need not equal ``own_l``: snapping a continuous L
    # onto an integer code moves it, and the sRGB round trip moves it again.
    # What it has to do is still decode to the code we asked for.
    assert abs(float(_ramp_code(_lab(blended)[0, 0, 0])) - code) <= 1


def test_encode_dab_preserves_hue_where_it_must_desaturate():
    """Desaturation scales a and b together, so the hue angle is kept.

    Breaks if the encoder clamps a and b independently to some range: that
    rotates the hue instead of washing it out, so a red pixel comes back
    orange rather than pink.
    """
    rgb = np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8)
    # A far L, which pure red cannot represent -- its own chroma is too high.
    depth_z = np.full(
        (2, 2),
        np.exp(np.log(Z_MIN) + 0.80 * np.log(Z_MAX / Z_MIN)),
        dtype=np.float32,
    )
    before = _lab(rgb)[0, 0, 1:]

    blended = encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX))

    after = _lab(blended)[0, 0, 1:]
    assert np.linalg.norm(after) < np.linalg.norm(before)  # it did desaturate
    assert np.degrees(
        abs(np.arctan2(after[1], after[0]) - np.arctan2(before[1], before[0]))
    ) < 2.0


def test_encode_dab_writes_black_for_invalid_pixels():
    """Invalid pixels pack to (0, 0, 0), the sentinel the packed image uses.

    Breaks if validity is ignored: an invalid pixel would then encode its
    placeholder depth and decode as a real but wrong one.
    """
    depth_z = _dab_depth()
    valid = np.zeros(depth_z.shape, dtype=bool)
    valid[:, : depth_z.shape[1] // 2] = True

    blended = encode_dab(_dab_rgb(), depth_z, valid, z_range=(Z_MIN, Z_MAX))

    assert (blended[~valid] == 0).all()


def test_dab_round_trip_keeps_invalid_pixels_invalid():
    """The sentinel survives the round trip in both directions."""
    depth_z = _dab_depth()
    valid = np.zeros(depth_z.shape, dtype=bool)
    valid[:, : depth_z.shape[1] // 2] = True

    _, back_valid = decode_dab(
        encode_dab(_dab_rgb(), depth_z, valid, z_range=(Z_MIN, Z_MAX)),
        Z_MIN,
        Z_MAX,
    )

    assert not back_valid[~valid].any()
    assert back_valid[valid].all()


def test_encode_dab_keeps_a_near_rail_pixel_valid():
    """A pixel sitting on the near rail stays valid rather than becoming the
    sentinel.

    Code 1 lands at L = 0.39, so the sRGB round trip only has to lose half a
    code for the pixel to read back as depth 0, i.e. invalid -- and an adaptive
    range puts the frame's own nearest surface there by construction, so this
    is not a corner case. The invalid sentinel is a *deep* zero, so it needs a
    real margin, not just a correct code.
    """
    rgb = _dab_rgb(8, 8)
    depth_z = np.full((8, 8), Z_MIN, dtype=np.float32)

    recovered, valid = decode_dab(
        encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0),
        Z_MIN,
        Z_MAX,
    )

    assert valid.all()
    assert np.allclose(recovered, Z_MIN, rtol=DAB_RTOL)


def test_encode_dab_clips_depths_below_the_range_to_the_near_bound():
    """A depth under the encoded range clips to the near bound, and survives.

    Breaks if clipping is left to the code cast alone, which turns an
    out-of-range pixel into code 0 and loses it to the invalid sentinel.
    """
    rgb = _dab_rgb(8, 8)
    depth_z = np.full((8, 8), Z_MIN / 4.0, dtype=np.float32)

    recovered, valid = decode_dab(
        encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0),
        Z_MIN,
        Z_MAX,
    )

    assert valid.all()
    assert np.allclose(recovered, Z_MIN, rtol=DAB_RTOL)


def test_encode_dab_only_washes_out_the_pixels_the_payload_needs():
    """The desaturation is targeted: most of the frame keeps its colour.

    Breaks if the encoder reduces chroma beyond what the payload needs. Making
    the predicate demand an *exact* code rather than a within-tolerance one, for
    instance, takes this fixture's washed-out share from 50% to 69% while
    decoding to an identical depth -- colour spent for no accuracy at all. The
    60% bound sits clear of both, and the share is deterministic, so it is a
    fixed number rather than a flaky one.

    Counted as a pixel whose chroma has dropped below 95% of its input chroma,
    ignoring near-neutral pixels where the ratio is mostly rounding noise.
    """
    rgb, depth_z = _dab_rgb(), _dab_depth()

    blended = encode_dab(rgb, depth_z, z_range=(Z_MIN, Z_MAX))

    before = np.linalg.norm(_lab(rgb)[..., 1:], axis=-1)
    after = np.linalg.norm(_lab(blended)[..., 1:], axis=-1)
    washed = (after < 0.95 * before) & (before > 1.0)

    assert washed.mean() < 0.60, (
        f"{washed.mean():.0%} of the frame lost chroma; the payload needs "
        f"about half of that"
    )


# ---------------------------------------------------------- Alb_normal
# The third carrier, and the only one with no depth in it: R carries an albedo
# map, G and B carry the normal's nx and ny.  Two channels for a normal is one
# short of what a sphere needs, so the hemisphere is resolved from the model's
# declared pole rather than stored -- which is lossy for the small share of
# pixels that sit on the far side of it.  See
# test_alb_normal_is_lossy_across_the_pole for what that costs, and note the
# tests below pin the loss rather than hiding it.

#: Away from the fold the pair is quantized over [-1, 1] at 8 bits, so one code
#: is 2/255 = 0.0078 and half a step is 0.0039 -- which, in radians, is 0.22
#: degrees of direction.  Asserting a hair above that pins the encoding to the
#: 8-bit floor rather than to "looks about right".
ALB_NORMAL_AWAY_FROM_FOLD_DEG = 0.5

#: What the fold costs.  nz comes back as sqrt(1 - nx^2 - ny^2), which is
#: infinitely steep as nx^2 + ny^2 approaches 1: an nx that rounds up to
#: exactly 1.0 forces nz to 0 and throws away whatever real nz there was.
#: Worst measured on the sweep below is 4.9 degrees, on a normal 84 degrees
#: from the pole, so 5 bounds it without being far off.
ALB_NORMAL_AT_FOLD_DEG = 5.0


def _albedo(height: int = 24, width: int = 24) -> np.ndarray:
    """Any byte values will do -- the albedo is carried verbatim -- but the
    fixture deliberately includes the 0 and 255 ends."""
    rng = np.random.default_rng(0)
    albedo = rng.integers(0, 256, (height, width), dtype=np.uint8)
    albedo[0, 0], albedo[0, 1] = 0, 255
    return albedo


def _alb_normal_normals(height: int = 24, width: int = 24,
                        nz_sign: float = -1.0) -> np.ndarray:
    """Unit normals sweeping one hemisphere, from just off the pole to 89
    degrees out.

    Swept in polar coordinates rather than as a square grid of (nx, ny): a
    grid's corners are outside the unit disk, so they are not normals at all
    and would quietly test an impossible input. The sweep stops at 89 degrees
    so it comes right up to the fold without sitting exactly on it, where the
    reconstruction's z is degenerate.
    """
    tilt = np.linspace(0.0, np.radians(89.0), height)[:, None]
    azimuth = np.linspace(0.0, 2.0 * np.pi, width, endpoint=False)[None, :]
    nx = np.sin(tilt) * np.cos(azimuth)
    ny = np.sin(tilt) * np.sin(azimuth)
    nz = nz_sign * np.cos(tilt)
    nx, ny, nz = np.broadcast_arrays(nx, ny, nz)
    return np.stack([nx, ny, nz], axis=-1).astype(np.float32)


def _decode_alb(image, pole=None):
    if pole is None:
        pole = MOGE_POLE
    return decode_alb_normal(image, pole=np.asarray(pole, dtype=np.float64))


def test_encode_alb_normal_returns_uint8_rgb():
    """The carrier is an (H, W, 3) uint8 RGB, like the other two."""
    image = encode_alb_normal(_albedo(4, 6), _alb_normal_normals(4, 6))

    assert image.shape == (4, 6, 3)
    assert image.dtype == np.uint8


def test_alb_normal_round_trip_recovers_the_albedo_exactly():
    """The albedo is stored, not encoded: it comes back byte for byte.

    Breaks if the albedo is rescaled, normalized or passed through the Lab
    round trip the depth carrier needs -- all of which would move it.
    """
    albedo = _albedo()

    image = encode_alb_normal(albedo, _alb_normal_normals())
    _, albedo_back, valid = _decode_alb(image)

    assert valid.all()
    assert np.array_equal(albedo_back[valid], albedo[valid])


def _angle_deg(a, b):
    """Angle between two (..., 3) fields, in degrees."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def test_alb_normal_round_trip_recovers_normals_to_quantization():
    """Normals in the pole's hemisphere come back to the 8-bit floor, except
    right at the fold where the reconstruction is ill conditioned.

    Breaks if the channel rescale, the channel order or the hemisphere sign is
    wrong -- all of which cost whole degrees everywhere, not fractions of one
    away from the fold.

    The two bands are the carrier's actual conditioning profile, not a
    tolerance picked to make it pass. Near the pole the error is the
    quantization half-step; approaching the fold it grows, because nz is
    recovered as a square root of 1 - nx^2 - ny^2 and that is infinitely steep
    there. Pinning both means a change that flattens the near-pole case, or
    that makes the fold case worse, shows up as a failure.
    """
    normals = _alb_normal_normals()

    back, _, valid = _decode_alb(encode_alb_normal(_albedo(), normals))

    err = _angle_deg(back, normals)
    tilt = np.degrees(np.arccos(np.clip(-normals[..., 2], -1.0, 1.0)))
    assert valid.all()
    assert err[tilt <= 60.0].max() <= ALB_NORMAL_AWAY_FROM_FOLD_DEG
    assert err.max() <= ALB_NORMAL_AT_FOLD_DEG


@pytest.mark.parametrize("pole, nz_sign", [((0, 0, -1), -1.0), ((0, 0, 1), 1.0)],
                         ids=["opencv", "opengl"])
def test_decode_alb_normal_resolves_the_hemisphere_from_the_pole(pole, nz_sign):
    """The missing z comes from the pole, not from a hard-coded sign.

    Two channels cannot say which side of the fold a normal is on, so the
    reader supplies it: the normal is taken to be on the pole's side. Breaks if
    the sign is baked in rather than read from the pole, which flips every
    normal for any model whose convention differs.
    """
    normals = _alb_normal_normals(nz_sign=nz_sign)

    back, _, _ = _decode_alb(encode_alb_normal(_albedo(), normals), pole=pole)

    assert _angle_deg(back, normals).max() <= ALB_NORMAL_AT_FOLD_DEG


def test_alb_normal_is_lossy_across_the_pole():
    """A normal behind the pole comes back as its mirror, and that is accepted.

    Two channels hold a hemisphere, so a pixel that really does point the other
    way is indistinguishable from its reflection -- there is nothing in the
    image to tell them apart. Measured over six real frames, 0.184% of pixels
    land here at more than 10 degrees out and the worst reaches 49. This test
    pins the behaviour so it stays a known, bounded cost rather than becoming a
    surprise; the octahedral carrier is the one to use when the whole sphere
    has to survive.
    """
    normals = _alb_normal_normals(nz_sign=+1.0)  # behind an OpenCV-frame pole

    back, _, _ = _decode_alb(encode_alb_normal(_albedo(), normals))

    mirrored = np.stack([normals[..., 0], normals[..., 1], -normals[..., 2]], -1)
    assert _angle_deg(back, mirrored).max() <= ALB_NORMAL_AT_FOLD_DEG
    # and the truth really is far away, by the angle between a normal and its
    # own reflection -- which is 2 * the tilt from the fold
    assert np.median(_angle_deg(back, normals)) > 30.0
    assert _angle_deg(back, normals).max() > 170.0


def test_decode_alb_normal_rejects_a_pole_in_the_xy_plane():
    """A pole with no z cannot say which side of the fold a normal is on.

    Silently guessing would flip arbitrary normals, so the carrier refuses:
    this encoding is only meaningful for a model whose normals are concentrated
    in one z-hemisphere, and a pole lying in the xy-plane says they are not.
    """
    with pytest.raises(ValueError, match="hemisphere"):
        _decode_alb(np.zeros((2, 2, 3), dtype=np.uint8), pole=(1.0, 0.0, 0.0))


def test_encode_alb_normal_writes_black_for_invalid_pixels():
    """Invalid pixels pack to (0, 0, 0), the sentinel the other carriers use."""
    valid = np.zeros((8, 8), dtype=bool)
    valid[:, :4] = True

    image = encode_alb_normal(_albedo(8, 8), _alb_normal_normals(8, 8), valid)

    assert (image[~valid] == 0).all()


def test_alb_normal_round_trip_keeps_invalid_pixels_invalid():
    """The sentinel survives in both directions."""
    valid = np.zeros((8, 8), dtype=bool)
    valid[:, :4] = True

    _, _, back_valid = _decode_alb(
        encode_alb_normal(_albedo(8, 8), _alb_normal_normals(8, 8), valid)
    )

    assert not back_valid[~valid].any()
    assert back_valid[valid].all()


def test_encode_alb_normal_saturates_a_non_unit_input_rather_than_wrapping():
    """An nx outside [-1, 1] saturates at the channel's end instead of wrapping.

    (1.5 + 1) * 0.5 * 255 is 319, and a uint8 cast turns that into 62 -- so a
    normal pointing hard to one side would come back at nx = -0.514, pointing
    the other way. Breaks if the clip is dropped; it is the encode side's only
    guard against a caller handing over a normal that is not unit.
    """
    normals = np.array([[[1.5, -2.0, 0.0]]], dtype=np.float32)

    image = encode_alb_normal(np.zeros((1, 1), dtype=np.uint8), normals)

    assert image[0, 0, 1] == 255  # nx saturates high
    assert image[0, 0, 2] == 0  # ny saturates low


def test_decode_alb_normal_returns_unit_normals_at_the_square_edge():
    """Pixels at the edge of the (nx, ny) square still decode to unit vectors.

    An nx of +1 with any nonzero ny puts nx^2 + ny^2 past 1, so the recovered z
    clamps to 0 and the raw vector comes out longer than unit. That is
    reachable in practice rather than only in principle -- the sweep fixture
    lands 20 of its pixels past 1. Breaks if the result is not re-normalized,
    and every consumer of these takes them as unit normals.
    """
    image = np.array([[[200, 255, 129], [200, 255, 255]]], dtype=np.uint8)

    normals, _, valid = _decode_alb(image)

    nx = image[..., 1].astype(np.float32) / 255.0 * 2.0 - 1.0
    ny = image[..., 2].astype(np.float32) / 255.0 * 2.0 - 1.0
    assert (nx**2 + ny**2).max() > 1.0  # the raw reconstruction is over-long
    assert valid.all()
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)


# ------------------------------------------------- Dab: the detail layer
# The depth channel is otherwise a smooth ramp, which makes the image read as a
# depth map rather than as a photograph. Adding back L's *high frequencies*
# restores the surface markings while leaving the slow lighting behind -- and
# simultaneously spends depth accuracy, because the markings and the payload
# share the one channel. These tests pin both halves of that trade.


def _ramp_code(l_star: np.ndarray) -> np.ndarray:
    """The depth code an L* sits at, walking the documented encode ramp.

    Recomputed from the formula rather than read off the encoder: the tests
    that use it are about the *scale* of what was added, so deriving the code
    from the implementation would let a change to the ramp cancel itself out.
    """
    along = (l_star - DAB_L_FAR) / (DAB_L_NEAR - DAB_L_FAR)
    return np.round(DEPTH_CODE_MAX - along * (DEPTH_CODE_MAX - DEPTH_CODE_MIN))


def _grey_frame(values: np.ndarray) -> np.ndarray:
    """An RGB frame whose Lab L is controlled by a grey level map."""
    return np.repeat(np.clip(values, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)


def _depth_error(blended: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Relative depth error of ``blended`` against a ``reference`` encoding."""
    z_ref, _ = decode_dab(reference, Z_MIN, Z_MAX)
    z_new, valid = decode_dab(blended, Z_MIN, Z_MAX)
    return (np.abs(z_new - z_ref) / z_ref)[valid]


def test_dab_adds_a_detail_layer_to_the_depth_channel():
    """The detail term changes the image, and only by the intended amount.

    Breaks if the detail is dropped (the image equals the plain depth encode)
    or added with the wrong scale.
    """
    texture = 128.0 + 30.0 * np.sin(np.arange(48) * 1.7)[None, :]
    frame = _grey_frame(np.repeat(texture, 24, axis=0))
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)

    plain = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)
    detailed = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX))

    assert not np.array_equal(plain, detailed)
    # the markings are a nudge on the depth code, not a rewrite of it
    assert _depth_error(detailed, plain).max() < 5 * ONE_CODE


def test_dab_strips_a_slow_lighting_gradient_from_the_depth():
    """A smooth brightness ramp must not move the depth at all.

    This is why the term is a high-pass rather than L itself: slow lighting is
    not surface detail, and a carrier that keeps it would trade depth accuracy
    for nothing. Breaks if the raw L is blended in, or if the bilateral's
    sigma_color grows until it stops tracking a gradient this slow.
    """
    ramp = np.tile(np.linspace(40.0, 220.0, 256), (64, 1))
    frame = _grey_frame(ramp)
    depth_z = np.full((64, 256), MID_Z, dtype=np.float32)

    plain = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)
    detailed = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX))

    assert _depth_error(detailed, plain).max() <= ONE_CODE * 1.05


def test_dab_detail_layer_can_be_switched_off():
    """``detail_alpha=0`` restores the exact carrier.

    It is the only guard against the detail being hard-wired, which would
    silently cost every reader the accuracy this carrier was built to keep.
    """
    frame = _grey_frame(np.tile(np.linspace(40.0, 220.0, 64), (16, 1)))
    depth_z = np.full((16, 64), MID_Z, dtype=np.float32)

    off = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)
    on = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.1)

    assert not np.array_equal(off, on)


def test_the_detail_is_added_at_the_documented_scale():
    """The markings are added at exactly ``alpha * (L - bilateral(L))``.

    Recomputed here from the formula rather than read off the implementation:
    a term added at the wrong scale still looks like texture, and still
    satisfies every test that only asks whether something changed. Breaks if
    the high-pass, the scale factor, or the clamp to the ramp's ends moves.
    """
    texture = 128.0 + 30.0 * np.sin(np.arange(48) * 1.7)[None, :]
    frame = _grey_frame(np.repeat(texture, 24, axis=0))
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)
    alpha = 0.1

    plain = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)
    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX),
                         detail_alpha=alpha)

    l_star = _lab(frame)[..., 0]
    detail = l_star - cv2.bilateralFilter(
        l_star / 100.0, d=DAB_DETAIL_D, sigmaColor=DAB_DETAIL_SIGMA_COLOR,
        sigmaSpace=DAB_DETAIL_SIGMA_SPACE,
    ) * 100.0
    want = np.clip(_lab(plain)[..., 0] + alpha * detail,
                   *sorted((DAB_L_FAR, DAB_L_NEAR)))

    # atol is in L* units; the uint8 round trip is worth a few tenths of one
    assert np.allclose(_lab(blended)[..., 0], want, atol=1.0)


def test_the_detail_high_pass_does_not_readd_strong_edges():
    """Across a hard edge the detail term stays near zero.

    That is the whole reason for a bilateral rather than a Gaussian: an
    edge-preserving filter leaves strong edges out of the residual, so the term
    costs far less depth accuracy for the same surface markings. On this step
    it pushes 0.25 codes through the edge where a Gaussian of the same spatial
    scale pushes 6.7. Breaks if the filter is swapped, or its sigmaColor is
    raised until it stops being edge preserving.
    """
    step = np.full((64, 64), 60.0)
    step[:, 32:] = 200.0
    frame = _grey_frame(step)
    depth_z = np.full((64, 64), MID_Z, dtype=np.float32)

    plain = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)
    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX))

    z_ref, _ = decode_dab(plain, Z_MIN, Z_MAX)
    z_new, _ = decode_dab(blended, Z_MIN, Z_MAX)
    codes = np.abs(z_new - z_ref) / z_ref / ONE_CODE

    assert codes.max() < 3.0


def test_the_detail_filter_parameters_are_threaded_through():
    """Each filter parameter actually reaches the bilateral.

    They are exposed so the trade can be tuned from outside the codec, and a
    parameter that stops being passed is silently a no-op -- the image looks
    fine and the knob does nothing. Breaks if any of the three is dropped
    between the signature and the filter call.
    """
    texture = 128.0 + 30.0 * np.sin(np.arange(48) * 1.7)[None, :]
    frame = _grey_frame(np.repeat(texture, 24, axis=0))
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)

    def encoded(**kwargs):
        return encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX),
                          detail_alpha=0.1, **kwargs)

    base = encoded()
    assert not np.array_equal(base, encoded(detail_d=3))
    assert not np.array_equal(base, encoded(detail_sigma_color=1.0))
    # a *tighter* sigma_space, not a looser one: with d capping the window
    # the loose end stops mattering well before the output can see it
    assert not np.array_equal(base, encoded(detail_sigma_space=1))


# ------------------------------------------------- Dab: the brightness ramp
# Depth is carried as luminance, so the ramp's direction and span are what the
# image looks like. Near surfaces are put at the bright end, as a photograph
# has them, and the span is compressed away from the ends of the luminance
# range -- at the full 0-100 the nearest object is near-black and the farthest
# is near-white, neither of which reads as a picture.


def test_dab_keeps_the_ramp_inside_the_gate():
    """No part of the ramp escapes the two ends it was given.

    What the gate is set to is the caller's business -- the tool's defaults
    keep it well off 0 and 100, but that is a tuning choice, not this codec's
    invariant. Breaks if the ramp is written over some other span, e.g. the
    full 0-100, which is what put the nearest surface at near-black and the
    farthest at near-white.
    """
    depth_z = np.linspace(Z_MIN, Z_MAX, 64, dtype=np.float32)[None, :]
    depth_z = np.repeat(depth_z, 8, axis=0)
    frame = _grey_frame(np.full((8, 64), 128.0))

    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)

    low, high = sorted((DAB_L_FAR, DAB_L_NEAR))
    l_star = _lab(blended)[..., 0]
    assert l_star.min() >= low - 1.0
    assert l_star.max() <= high + 1.0


def test_decode_dab_honours_the_l_range_it_is_given():
    """A reader that passes the writer's range recovers the depth; one that
    guesses the defaults does not.

    The range is a property of the encoding, not of the codec, so it has to
    travel with the image -- which is why the tool records it in the sidecar.
    Breaks if decode stops reading the range, at which point a tuned encode
    decodes silently wrong rather than failing.
    """
    depth_z = np.linspace(Z_MIN, Z_MAX, 32, dtype=np.float32)[None, :]
    depth_z = np.repeat(depth_z, 8, axis=0)
    frame = _grey_frame(np.full((8, 32), 128.0))
    l_far, l_near = 30.0, 90.0

    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0,
                         l_far=l_far, l_near=l_near)

    told, _ = decode_dab(blended, Z_MIN, Z_MAX, l_far=l_far, l_near=l_near)
    guessed, _ = decode_dab(blended, Z_MIN, Z_MAX)

    assert np.allclose(told, depth_z, rtol=2 * DEPTH_RTOL)
    assert not np.allclose(guessed, depth_z, rtol=2 * DEPTH_RTOL)


def test_dab_puts_near_surfaces_at_the_dark_end():
    """The nearest surface is the darkest and the farthest the brightest.

    Breaks if the ramp runs the other way -- which it has, in both directions,
    so what this pins is that the ends stay named for the ends rather than for
    which one happens to be bright.
    """
    depth_z = np.array([[Z_MIN, MID_Z, Z_MAX]], dtype=np.float32)
    frame = _grey_frame(np.full((1, 3), 128.0))

    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX), detail_alpha=0.0)

    l_star = _lab(blended)[..., 0]
    assert l_star[0, 0] < l_star[0, 1] < l_star[0, 2]
    assert l_star[0, 0] == pytest.approx(DAB_L_NEAR, abs=1.0)
    assert l_star[0, 2] == pytest.approx(DAB_L_FAR, abs=1.0)


@pytest.mark.parametrize("l_far, l_near", [(85.0, 20.0), (20.0, 85.0)],
                         ids=["far bright", "near bright"])
def test_the_detail_is_clamped_to_the_ramp_whichever_way_it_runs(l_far, l_near):
    """The detail cannot push a pixel past either end, in either direction.

    The clamp is a min/max on L, and the ramp's ends are *not* in ascending
    order once the far end is the bright one -- so a clamp written the obvious
    way round, ``clip(x, l_far, l_near)``, silently collapses the whole image
    to a single value. Breaks if the clamp stops sorting its two ends.
    """
    texture = 128.0 + 30.0 * np.sin(np.arange(48) * 1.7)[None, :]
    frame = _grey_frame(np.repeat(texture, 24, axis=0))
    depth_z = np.linspace(Z_MIN, Z_MAX, 48, dtype=np.float32)[None, :]
    depth_z = np.repeat(depth_z, 24, axis=0)

    blended = encode_dab(frame, depth_z, z_range=(Z_MIN, Z_MAX),
                         l_far=l_far, l_near=l_near)

    l_star = _lab(blended)[..., 0]
    low, high = sorted((l_far, l_near))
    assert l_star.min() >= low - 0.5
    assert l_star.max() <= high + 0.5
    # and it has not been flattened into one value by a backwards clamp
    assert l_star.max() - l_star.min() > 5.0


def test_the_decoder_never_gives_a_non_zero_pixel_the_zero_code():
    """A pixel that is not the sentinel always comes back with a depth.

    Hand-built rather than encoded, because the value that matters is one just
    past the ramp's dark end and the encoder only reaches it when the gate is
    narrow enough that a fraction of an L* is worth a code. A valid pixel whose
    L round-trips there lands below code 1, and code 0 reads as "no depth" to
    ``_depth_decode`` -- so the reader is told nothing about a pixel the
    sentinel test, which looks at the pixel being all-zero, calls valid. The
    two have to agree about every pixel. At gate 50-1, grey 3 sits at L* 0.82
    and a raw code of 0.06. Breaks if the recovered code is let back below 1.
    """
    image = np.full((1, 2, 3), 3, dtype=np.uint8)  # dark, but not the sentinel

    depth, valid = decode_dab(image, Z_MIN, Z_MAX, l_far=50.0, l_near=1.0)

    assert valid.all()
    assert (depth > 0.0).all()


# ------------------------------------------------------------ L_normal
# The frame's own luminance in L, the normal's nx and ny in a and b.  Where
# Alb_normal spends R on an albedo, this one spends nothing extra -- but it
# puts the payload in the worst place in Lab for it.  a and b are signed chroma
# and the sRGB gamut only admits so much of them at a given luminance, so the
# capacity is set by each pixel's own L: wide at mid grey, nearly nothing at
# either end.  Measured on a real frame, the frame's L gives +-80 in a and +-56
# in b at L* 50, but +-8 in a at L* 95.  These tests pin both the carrier's
# useful half and its lossy one.


def _l_normal_frame(level: float, height: int = 24, width: int = 48) -> np.ndarray:
    """A flat grey frame at a given level, so its Lab L is controlled."""
    return _grey_frame(np.full((height, width), level))


def _l_normal_error(frame, normals, scale=None, l_far=None):
    kwargs = {} if scale is None else {"scale": scale}
    image = encode_l_normal(frame, normals, **kwargs)
    back, valid = decode_l_normal(image, pole=np.asarray(MOGE_POLE), **kwargs)
    assert valid.all()
    return np.degrees(np.arccos(np.clip(np.sum(back * normals, -1), -1.0, 1.0)))


def test_l_normal_keeps_the_frames_own_luminance():
    """The L channel comes back as the luminance that went in.

    This is the whole of what makes it different from --save-Alb-norm, whose R
    channel is a Retinex albedo instead. Breaks if L is overwritten with the
    depth, the albedo, or anything else.
    """
    frame = _grey_frame(np.tile(np.linspace(30.0, 210.0, 48), (24, 1)))
    before = _lab(frame)[..., 0]

    image = encode_l_normal(frame, _alb_normal_normals(24, 48))

    after = _lab(image)[..., 0]
    # a few L* of slack where the chroma runs into the gamut and drags L with it
    assert np.abs(after - before).max() < 6.0


def test_l_normal_round_trips_normals_where_the_gamut_allows():
    """At mid luminance the gamut is wide enough to carry the pair.

    Breaks if the channel order, the scale or the hemisphere sign is wrong --
    all of which cost whole tens of degrees, not the single digits this allows.
    """
    normals = _alb_normal_normals(24, 48)

    err = _l_normal_error(_l_normal_frame(128.0), normals)

    assert np.median(err) < 4.0


def test_l_normal_loses_normals_where_the_gamut_is_tight():
    """At high luminance the same normals come back much worse.

    This is not a bug to fix, it is the container: a and b are gamut-limited,
    and at L* 95 the sRGB gamut admits about +-8 of chroma against the +-56
    this scale needs. Pinned so the loss stays a known, measured property
    rather than something a future change quietly widens -- and so nobody
    reaches for this carrier expecting Alb_normal's accuracy.
    """
    normals = _alb_normal_normals(24, 48)

    mid = np.median(_l_normal_error(_l_normal_frame(128.0), normals))
    bright = np.median(_l_normal_error(_l_normal_frame(235.0), normals))

    assert bright > 4 * mid


def test_decode_l_normal_honours_the_scale_it_is_given():
    """A reader that passes the writer's scale recovers the normals; the
    default does not.

    Breaks if decode stops reading the scale, at which point a retuned encode
    decodes silently wrong rather than failing.
    """
    normals = _alb_normal_normals(24, 48)
    frame = _l_normal_frame(128.0)
    scale = 30.0

    image = encode_l_normal(frame, normals, scale=scale)

    told, _ = decode_l_normal(image, pole=np.asarray(MOGE_POLE), scale=scale)
    guessed, _ = decode_l_normal(image, pole=np.asarray(MOGE_POLE))

    def err(a):
        return np.degrees(np.arccos(np.clip(np.sum(a * normals, -1), -1.0, 1.0)))

    assert np.median(err(told)) < 4.0
    assert np.median(err(guessed)) > 10.0


@pytest.mark.parametrize("pole, nz_sign", [((0, 0, -1), -1.0), ((0, 0, 1), 1.0)],
                         ids=["opencv", "opengl"])
def test_decode_l_normal_resolves_the_hemisphere_from_the_pole(pole, nz_sign):
    """As with the other two-channel carrier, the missing z comes from the pole."""
    normals = _alb_normal_normals(24, 48, nz_sign=nz_sign)
    frame = _l_normal_frame(128.0)

    image = encode_l_normal(frame, normals)
    back, _ = decode_l_normal(image, pole=np.asarray(pole, dtype=np.float64))

    err = np.degrees(np.arccos(np.clip(np.sum(back * normals, -1), -1.0, 1.0)))
    assert np.median(err) < 4.0


def test_decode_l_normal_rejects_a_pole_in_the_xy_plane():
    """A pole with no z cannot say which side of the fold a normal is on."""
    with pytest.raises(ValueError, match="hemisphere"):
        decode_l_normal(np.zeros((2, 2, 3), dtype=np.uint8),
                        pole=np.array([1.0, 0.0, 0.0]))


def test_l_normal_round_trip_keeps_invalid_pixels_invalid():
    """Invalid pixels pack to (0, 0, 0) and stay invalid through the trip."""
    valid = np.zeros((24, 48), dtype=bool)
    valid[:, :24] = True

    image = encode_l_normal(_l_normal_frame(128.0), _alb_normal_normals(24, 48),
                            valid)

    _, back_valid = decode_l_normal(image, pole=np.asarray(MOGE_POLE))
    assert (image[~valid] == 0).all()
    assert not back_valid[~valid].any()
    assert back_valid[valid].all()


def test_decode_l_normal_returns_unit_normals_at_the_square_edge():
    """A pixel whose pair runs past the unit circle still decodes to a unit
    vector.

    The image can hold (nx, ny) = (1, 1), which no unit normal is, and rounding
    can push a real one past 1 as well. The recovered z clamps to 0 there and
    the raw vector comes out longer than unit, so it has to be renormalized
    before it reaches anything that treats these as directions. Breaks if the
    decode stops normalizing.
    """
    lab = np.array([[[50.0, 56.0, 56.0]]], dtype=np.float32)
    rgb = (np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)
           * 255).round().astype(np.uint8)

    normals, valid = decode_l_normal(rgb, pole=np.asarray(MOGE_POLE))

    assert valid.all()
    assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-6)


# --------------------------------------------------------------- fused
# The packed image with the depth channel carrying an L* detail term on top of
# the code: [Oct_U, Oct_V, d + alpha*R_L*].  The layout is deliberately the
# same as the packed image's, so the detail is the *only* difference and the
# existing decoder reads it without knowing this carrier exists.

def _fused_frame(height: int = 24, width: int = 48) -> np.ndarray:
    """A frame with real fine texture, so the detail term is not ~0.

    A smooth ramp will not do: the detail is the *high* frequencies of L, and
    a gradient has none, so a smooth fixture leaves the term near zero and
    every test built on it passes for the wrong reason.
    """
    return _grey_frame(np.tile(128.0 + 40.0 * np.sin(np.arange(width) * 1.7),
                               (height, 1)))


def _fused(normals, depth_z, frame, alpha=0.05, valid=None,
           z_range=(Z_MIN, Z_MAX), pack=CODEC):
    return encode_fused(normals, depth_z, frame, valid, z_range=z_range,
                        detail_alpha=alpha, pack=pack)


def test_encode_fused_returns_uint8_rgb():
    """The fused image is an (H, W, 3) uint8 RGB, like the packed one."""
    normals = _alb_normal_normals(4, 6)

    image = _fused(normals, np.full((4, 6), MID_Z, dtype=np.float32),
                   _fused_frame(4, 6))

    assert image.shape == (4, 6, 3)
    assert image.dtype == np.uint8


def test_fused_leaves_the_normal_channels_byte_identical_to_the_packed_image():
    """Only the depth channel differs from the packed encoding.

    The point of matching the packed layout is that the detail is the whole of
    the difference. Breaks if the detail leaks into the octahedral pair -- by a
    channel mix-up, or by the detail being added before the projection rather
    than after it.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)

    packed = CODEC.encode(normals, depth_z, z_range=(Z_MIN, Z_MAX))
    fused = _fused(normals, depth_z, _fused_frame())

    assert np.array_equal(fused[..., :2], packed[..., :2])
    assert not np.array_equal(fused[..., 2], packed[..., 2])


def test_fused_reads_back_with_the_existing_decoder():
    """``pack.decode`` recovers the normals exactly and the depth nearly.

    That is the whole reason for matching the packed layout: no second decoder
    and no second vocabulary. The depth comes back with the detail as its
    error, which is the trade this carrier makes and the packed image does not.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)

    fused = _fused(normals, depth_z, _fused_frame())

    back, back_z, valid = _decode(fused)
    assert valid.all()
    assert np.allclose(back, normals, atol=0.05)
    rel = np.abs(back_z - depth_z) / depth_z
    assert rel.max() < 0.05


def test_fused_detail_is_scaled_like_the_l_carriers():
    """The term lands on the depth channel at the documented strength.

    Recomputed here from the formula rather than read off the implementation,
    because the units are the whole game: a code is DAB_L_PER_CODE = 0.39 L*,
    so a term that forgets to convert is 2.55x too strong -- the same mistake
    encode_dab's spec test exists to catch, and it looks identical from the
    outside.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)
    frame = _fused_frame()
    alpha = 0.08

    fused = _fused(normals, depth_z, frame, alpha=alpha)
    plain = _fused(normals, depth_z, frame, alpha=0.0)

    l_star = _lab(frame)[..., 0]
    detail = l_star - cv2.bilateralFilter(
        l_star / 100.0, d=DAB_DETAIL_D, sigmaColor=DAB_DETAIL_SIGMA_COLOR,
        sigmaSpace=DAB_DETAIL_SIGMA_SPACE,
    ) * 100.0
    want = alpha * detail / DAB_L_PER_CODE

    got = (fused[..., 2].astype(np.float32)
           - plain[..., 2].astype(np.float32))
    assert np.allclose(got, np.round(want), atol=1.0)


def test_fused_with_no_detail_is_the_packed_image():
    """``detail_alpha=0`` reproduces the packed encoding byte for byte.

    Breaks if the detail is hard-wired, or if the fused path perturbs the
    channel for some other reason -- which would make the two carriers differ
    by more than the one term this one exists to add.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)

    packed = CODEC.encode(normals, depth_z, z_range=(Z_MIN, Z_MAX))
    fused = _fused(normals, depth_z, _fused_frame(), alpha=0.0)

    assert np.array_equal(fused, packed)


def test_fused_keeps_invalid_pixels_on_the_zero_sentinel():
    """An invalid pixel stays all-zero, detail or not.

    The packed image marks invalid by zeroing the whole pixel, and the detail
    is added to that same channel -- so adding to a zero code lifts it to a
    small but real one, and the reader hands back a perfectly plausible depth
    for a pixel that has none. Breaks if the detail is not gated on the code
    being real.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)
    valid = np.zeros((24, 48), dtype=bool)
    valid[:, :24] = True

    fused = _fused(normals, depth_z, _fused_frame(), alpha=0.2, valid=valid)

    assert (fused[~valid] == 0).all()
    _, back_z, back_valid = _decode(fused)
    assert not back_valid[~valid].any()


def test_fused_keeps_the_depth_channel_inside_the_valid_codes():
    """The detail cannot push a pixel out of the codes 1..255.

    Code 0 is the invalid sentinel, so a pixel pushed onto it loses its depth
    outright rather than clipping to a bound. Breaks if the detail is added
    without holding the code inside the valid range. Needs a strong alpha and a
    ramp that starts and ends on the rails, which is where clipping bites.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.exp(np.linspace(np.log(Z_MIN), np.log(Z_MAX), 48))
    depth_z = np.repeat(depth_z[None, :], 24, axis=0).astype(np.float32)

    fused = _fused(normals, depth_z, _fused_frame(), alpha=0.5)

    assert (fused[..., 2] > 0).all()


def test_fused_uses_the_codec_it_is_given():
    """The caller's codec supplies the pole, and a different pole differs.

    Breaks if ``pack`` is ignored and a default is built instead: the octahedral
    pair would come out for a pole the caller did not ask for, which decodes to
    normals rotated by the difference and looks entirely plausible.
    """
    normals = _alb_normal_normals(24, 48)
    depth_z = np.full((24, 48), MID_Z, dtype=np.float32)
    frame = _fused_frame()

    down = _fused(normals, depth_z, frame,
                  pack=NormalDepthPack(pole=np.array([0.0, 0.0, -1.0]),
                                       check_conditioning=False))
    up = _fused(normals, depth_z, frame,
                pack=NormalDepthPack(pole=np.array([0.0, 0.0, 1.0]),
                                     check_conditioning=False))

    assert not np.array_equal(down[..., :2], up[..., :2])
