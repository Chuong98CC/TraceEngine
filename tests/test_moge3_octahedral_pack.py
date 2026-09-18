"""Octahedral normal + logz packing from the MoGe v3 tool.

The packing helpers are pure numpy, so they run on CPU — unlike the tool's
``main``, which requires CUDA. Importing the tool does pull in the torch/MoGe
runtime, so these are not dependency-free tests.
"""

import numpy as np

from tools.general_test.module.infer_moge3 import (
    adaptive_logz_range,
    pack_octahedral_logz,
    unpack_octahedral_logz,
)

#: Bounds the depth channel is encoded over, in affine-z metres.
Z_MIN, Z_MAX = 0.25, 3.0

#: Codec quantization: 254 levels span log(3.0) - log(0.25) = 2.4849 nats, so
#: one level is 0.009783 nats -> 0.983% relative in z. Hand-derived from the
#: range, not read off the codec, with margin for the floor.
LOGZ_RTOL = 0.012

#: A mid-range affine z, used whenever the depth channel is not the subject.
MID_Z = 1.0


def _unit(normals) -> np.ndarray:
    """Normalize last-axis vectors (hand-built fixtures -> unit normals)."""
    normals = np.asarray(normals, dtype=np.float32)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _pack(normals, logz=None, valid=None, z_min=Z_MIN, z_max=Z_MAX) -> np.ndarray:
    """Pack with a mid-range logz unless one is given."""
    normals = np.asarray(normals, dtype=np.float32)
    if logz is None:
        logz = np.full(normals.shape[:2], np.log(MID_Z), dtype=np.float32)
    return pack_octahedral_logz(normals, np.asarray(logz), valid, z_min, z_max)


def test_pack_octahedral_shape_and_dtype():
    """The packed image is an (H, W, 3) uint8 RGB."""
    packed = _pack(_unit([[0, 0, 1], [1, 0, 0]]).reshape(1, 2, 3))

    assert packed.shape == (1, 2, 3)
    assert packed.dtype == np.uint8


def test_pack_octahedral_axis_normals_match_hand_derived_bytes():
    """Axis normals land on their hand-derived octahedral cells.

    Breaks if the L1 normalization, the [-1, 1] -> [0, 255] rescale, the
    channel order (U, V, depth) or the projection pole changes.
    """
    normals = np.array(
        [[[0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float32
    )

    packed = _pack(normals)

    # u = (px + 1) / 2 * 255, truncated to uint8. The centre is the
    # camera-facing normal, the corner is the one pointing away.
    assert packed[0, 0, :2].tolist() == [127, 127]  # -z (camera-facing) -> centre
    assert packed[0, 1, :2].tolist() == [255, 127]  # +x -> u max
    assert packed[0, 2, :2].tolist() == [127, 255]  # +y -> v max
    assert packed[0, 3, :2].tolist() == [255, 255]  # +z -> far corner


def test_pack_octahedral_puts_moges_camera_facing_normal_at_the_centre():
    """``(0, 0, -1)`` — MoGe's normal for a fronto-parallel surface — is the centre.

    MoGe emits normals in the OpenCV camera frame, so a surface facing the
    camera has nz = -1 (measured mean normal ``(0.04, -0.06, -1.00)`` across
    six cameras with unrelated viewpoints — a convention, not a scene
    property). The octahedral map is well conditioned at its pole and
    *degenerate* at the opposite one, so the projection pole has to sit on the
    data. This is that pole being in the right place.

    Breaks if the projection goes back to being about +z, which puts the whole
    scene on the degenerate far pole.
    """
    packed = _pack(np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32))

    assert packed[0, 0, :2].tolist() == [127, 127]


def test_pack_octahedral_keeps_a_flat_patch_tightly_clustered():
    """Normals within half a degree of each other must pack close together.

    ``(0, 0, -1)`` carries the bulk of MoGe's output — every fronto-parallel
    surface lands on it. Under the pre-fix pole it was the octahedral's
    four-way degenerate corner, where a hair of +nx versus -nx sends two
    neighbouring pixels of one flat surface to opposite corners of the square:
    measured at a 359-code jump between pixels whose normals differ by 0.56
    degrees. In the rendered image that is a hard-edged blotch on every flat
    wall.

    Breaks if the pole moves back off the data, however the pole is spelled.
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

    codes = _pack(normals.reshape(1, -1, 3))[0, :, :2].astype(np.float32)

    spread = float(np.linalg.norm(codes[:, None, :] - codes[None, :, :], axis=-1).max())
    assert spread < 20, (
        f"a 0.5 degree patch spread {spread:.0f} codes — the patch is sitting "
        f"on a degenerate point of the projection"
    )


def test_unpack_octahedral_decodes_hand_built_bytes():
    """Hand-built octahedral bytes decode to their hand-derived normals.

    Built from literals rather than by round-tripping through ``pack``, so it
    is the unpack side's own channel order and rescale under test. Breaks if
    the U/V channels are read in the wrong order — swapping them is invisible
    to a round-trip, which just applies the same swap twice.
    """
    # u = 255 -> +1, v = 127 -> ~0, so the first pixel is the +x normal. The
    # depth channel is a valid mid-range code, not the invalid 0.
    packed = np.array(
        [[[255, 127, 128], [127, 255, 128], [127, 127, 128]]], dtype=np.uint8
    )

    normals, _, valid = unpack_octahedral_logz(packed, Z_MIN, Z_MAX)

    assert np.allclose(normals[0, 0], [1, 0, 0], atol=0.05)  # u max -> +x
    assert np.allclose(normals[0, 1], [0, 1, 0], atol=0.05)  # v max -> +y
    assert np.allclose(normals[0, 2], [0, 0, -1], atol=0.05)  # centre -> -z
    assert valid.all()


def test_pack_octahedral_round_trips_back_facing_normals():
    """Normals pointing away from the camera survive pack -> unpack.

    These are the ones the projection has to fold: the pole sits on MoGe's
    camera-facing direction, so a normal pointing away from the camera is on
    the far side of it.

    Breaks if the folding scales by ``np.sign`` of the raw coordinate: sign(0)
    is 0, so a coordinate that is exactly 0 zeroes the other folded coordinate
    and the normal comes back rotated by up to 90 degrees.
    """
    normals = _unit([[[0, 0, 1], [0.001, 0, 0.999], [0, 0.001, 0.999]]])

    decoded, _, _ = unpack_octahedral_logz(_pack(normals), Z_MIN, Z_MAX)

    assert np.allclose(decoded, normals, atol=0.05)


def test_pack_octahedral_round_trips_random_unit_normals():
    """Both hemispheres survive pack -> unpack within 8-bit octahedral error."""
    rng = np.random.default_rng(0)
    normals = _unit(rng.normal(size=(32, 48, 3)))

    decoded, _, _ = unpack_octahedral_logz(_pack(normals), Z_MIN, Z_MAX)

    assert np.allclose(decoded, normals, atol=0.05)


def test_pack_octahedral_invalid_pixels_are_zero():
    """Pixels outside the mask pack to (0, 0, 0) in every channel."""
    normals = np.zeros((1, 2, 3), dtype=np.float32)
    normals[0, 0] = [0, 0, 1]
    valid = np.array([[True, False]])
    logz = np.array([[np.log(MID_Z), np.log(MID_Z)]], dtype=np.float32)

    packed = _pack(normals, logz, valid)

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_pack_octahedral_masks_pixels_whose_value_alone_looks_valid():
    """A ``valid=False`` pixel packs to (0, 0, 0) even with an in-range logz.

    Breaks if the ``valid`` argument is ignored. The logz here is finite and
    mid-range on both pixels, so the packer's own finiteness guard cannot
    catch it — the mask is the only thing that can.
    """
    normals = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    logz = np.full((1, 2), np.log(MID_Z), dtype=np.float32)

    packed = _pack(normals, logz, np.array([[True, False]]))

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_pack_octahedral_reserves_code_zero_for_invalid():
    """A valid pixel sitting exactly on ``z_min`` encodes to code 1, not 0.

    Code 0 is the invalid sentinel, so mapping the data onto [0, 255] would
    make every frame's *nearest* surfaces decode as masked-out. Breaks if the
    encoding loses the +1 offset.
    """
    normals = np.array([[[0, 0, 1]]], dtype=np.float32)
    logz = np.array([[np.log(Z_MIN)]], dtype=np.float32)

    packed = _pack(normals, logz)

    assert packed[0, 0, 2] == 1
    _, _, valid = unpack_octahedral_logz(packed, Z_MIN, Z_MAX)
    assert valid.all(), "a valid pixel at z_min must not decode as invalid"


def test_pack_octahedral_clips_out_of_range_to_the_range_ends():
    """Valid values beyond the range clip to codes 1 / 255 and stay valid.

    A user bound that is tighter than the scene must degrade a pixel to the
    range end, never to the invalid sentinel.
    """
    normals = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    logz = np.array([[np.log(Z_MIN / 4), np.log(Z_MAX * 4)]], dtype=np.float32)

    packed = _pack(normals, logz)
    _, decoded_logz, valid = unpack_octahedral_logz(packed, Z_MIN, Z_MAX)

    assert packed[0, 0, 2] == 1
    assert packed[0, 1, 2] == 255
    assert valid.all()
    assert np.allclose(np.exp(decoded_logz[0]),
                       [Z_MIN, Z_MAX], rtol=LOGZ_RTOL)


def test_pack_octahedral_round_trips_logz():
    """The depth channel round-trips affine z within the codec's bound."""
    z = np.array([[0.3, 0.5, 1.0, 2.5]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (1, 4, 1))

    _, decoded_logz, valid = unpack_octahedral_logz(
        _pack(normals, np.log(z)), Z_MIN, Z_MAX
    )

    assert valid.all()
    assert np.allclose(np.exp(decoded_logz), z, rtol=LOGZ_RTOL)


def test_pack_octahedral_channel_is_log_not_linear():
    """The geometric mid-range sits mid-scale: the channel is logarithmic.

    Breaks if the depth channel is swapped for a plain linear (z - min) /
    (max - min) normalization, which would put this z at code 58.
    """
    geometric_mid = np.sqrt(Z_MIN * Z_MAX)  # 0.866
    normals = np.array([[[0, 0, 1]]], dtype=np.float32)

    packed = _pack(normals, np.array([[np.log(geometric_mid)]], dtype=np.float32))

    # norm = 0.5 exactly -> 1 + 0.5 * 254 = 128.
    assert 127 <= packed[0, 0, 2] <= 129


def test_unpack_logz_pins_the_range_ends():
    """Codes 1 and 255 decode to exactly z_min and z_max.

    Pins the decode scaling, which the round-trip cannot: decoding over 255
    levels instead of 254 mis-scales by ~0.4%, and at the far end that is
    ~0.9% — inside the round-trip's tolerance. Breaks if the level count on
    either side of the packed image changes.
    """
    packed = np.array([[[127, 127, 1], [127, 127, 255]]], dtype=np.uint8)

    _, logz, valid = unpack_octahedral_logz(packed, Z_MIN, Z_MAX)

    assert valid.all()
    assert np.allclose(np.exp(logz[0]), [Z_MIN, Z_MAX], rtol=1e-6)


def test_unpack_logz_reconstructs_metric_depth():
    """packed + shift + metric_scale rebuilds the metric depth.

    This is the whole point of storing logz: the sidecar's two scalars turn
    the packed channel back into metres. Uses real per-frame values.
    """
    shift, metric_scale = -0.1382, 0.8784
    z = np.array([[0.689, 1.500, 2.753]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (1, 3, 1))

    _, decoded_logz, valid = unpack_octahedral_logz(
        _pack(normals, np.log(z)), Z_MIN, Z_MAX
    )
    metric = (np.exp(decoded_logz) + shift) * metric_scale

    assert valid.all()
    assert np.allclose(metric, (z + shift) * metric_scale, rtol=LOGZ_RTOL)


def test_adaptive_logz_range_tightens_to_the_data():
    """The frame's own affine-z range narrows the caller's bounds.

    Compared with a tolerance: the bounds come back as ``exp(logz)``, and
    ``exp(log(x))`` round-trips through float32 at ~1e-9, not exactly.
    """
    logz = np.log(np.array([[1.0, 2.0]], dtype=np.float32))

    lo, hi = adaptive_logz_range(logz, None, Z_MIN, Z_MAX)

    assert np.allclose([lo, hi], [1.0, 2.0], rtol=1e-6)


def test_adaptive_logz_range_keeps_the_bounds_when_they_are_tighter():
    """Bounds inside the data range win — they are a hard rail."""
    logz = np.log(np.array([[0.1, 10.0]], dtype=np.float32))

    assert adaptive_logz_range(logz, None, Z_MIN, Z_MAX) == (Z_MIN, Z_MAX)


def test_adaptive_logz_range_keeps_the_bounds_when_disjoint():
    """Data entirely outside the bounds falls back rather than inverting.

    Without the guard ``max(min) > min(max)`` and the range would be empty.
    """
    logz = np.log(np.array([[5.0, 6.0]], dtype=np.float32))

    assert adaptive_logz_range(logz, None, Z_MIN, Z_MAX) == (Z_MIN, Z_MAX)


def test_adaptive_logz_range_keeps_the_bounds_without_valid_pixels():
    """No valid pixels means nothing to tighten with."""
    logz = np.log(np.array([[1.0, 2.0]], dtype=np.float32))
    valid = np.array([[False, False]])

    assert adaptive_logz_range(logz, valid, Z_MIN, Z_MAX) == (Z_MIN, Z_MAX)


def test_adaptive_logz_range_ignores_non_finite_pixels():
    """A -inf placeholder must not drag the range down to zero.

    The tool leaves -inf where a pixel is invalid, so without the finiteness
    guard the data minimum would be ``exp(-inf) = 0`` and the range would
    widen back out to the caller's bounds, silently undoing the tightening.
    ``valid`` is None here so only the guard can catch it.
    """
    logz = np.array([[np.log(1.0), -np.inf, np.log(2.0)]], dtype=np.float32)

    lo, hi = adaptive_logz_range(logz, None, Z_MIN, Z_MAX)

    assert np.allclose([lo, hi], [1.0, 2.0], rtol=1e-6)


def test_adaptive_logz_range_ignores_pixels_marked_invalid():
    """A finite value at an invalid pixel must not widen the range either.

    Breaks if the ``valid`` argument is dropped from the intersection.
    """
    logz = np.array([[np.log(1.0), np.log(2.0), np.log(2.5)]], dtype=np.float32)
    valid = np.array([[True, False, True]])

    lo, hi = adaptive_logz_range(logz, valid, Z_MIN, Z_MAX)

    assert np.allclose([lo, hi], [1.0, 2.5], rtol=1e-6)


def test_unpack_logz_marks_code_zero_invalid():
    """Code 0 unpacks as valid=False, everything else as True.

    Breaks if the validity test is loosened to ``code >= 0``, which on a uint8
    is always true — every masked-out pixel would then read as a real surface
    at z = 1 m.
    """
    packed = np.array([[[127, 127, 0], [127, 127, 1]]], dtype=np.uint8)

    _, _, valid = unpack_octahedral_logz(packed, Z_MIN, Z_MAX)

    assert valid.tolist() == [[False, True]]
