"""Octahedral + log-depth packing from the MoGe v3 tool.

The packing helpers are pure numpy, so they run on CPU — unlike the tool's
``main``, which requires CUDA. Importing the tool does pull in the torch/MoGe
runtime, so these are not dependency-free tests.
"""

import numpy as np

from tools.general_test.module.infer_moge3 import (
    pack_octahedral_depth,
    unpack_octahedral_depth,
)

#: Depth every case sits at when the depth channel is not the subject: inside
#: the [MIN_DEPTH, MAX_DEPTH] log range, so it never clips.
MID_DEPTH = 0.5

#: Codec quantization: the log range spans log(1.251) - log(0.251) over 255
#: levels, so one level is 0.00630 in log space -> 0.63% relative in metres.
#: Hand-derived, not read off the codec, with margin for the round-half-down.
DEPTH_RTOL = 0.01


def _unit(normals) -> np.ndarray:
    """Normalize last-axis vectors (hand-built fixtures -> unit normals)."""
    normals = np.asarray(normals, dtype=np.float32)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _pack(normals, depth=None, valid=None) -> np.ndarray:
    """Pack with a uniform mid-range depth unless one is given.

    ``depth`` keeps its dtype: casting here would hide the codec's uint16-mm
    auto-detection from the tests below.
    """
    normals = np.asarray(normals, dtype=np.float32)
    if depth is None:
        depth = np.full(normals.shape[:2], MID_DEPTH, dtype=np.float32)
    return pack_octahedral_depth(normals, np.asarray(depth), valid)


def test_pack_octahedral_shape_and_dtype():
    """The packed image is an (H, W, 3) uint8 RGB."""
    packed = _pack(_unit([[0, 0, 1], [1, 0, 0]]).reshape(1, 2, 3))

    assert packed.shape == (1, 2, 3)
    assert packed.dtype == np.uint8


def test_pack_octahedral_axis_normals_match_hand_derived_bytes():
    """Forward/right/up normals land on their hand-derived octahedral cells.

    Breaks if the L1 normalization, the [-1, 1] -> [0, 255] rescale or the
    channel order (U, V, depth) changes.
    """
    normals = np.array([[[0, 0, 1], [1, 0, 0], [0, 1, 0]]], dtype=np.float32)

    packed = _pack(normals)

    # u = (px + 1) / 2 * 255, v = (py + 1) / 2 * 255, truncated to uint8.
    assert packed[0, 0, :2].tolist() == [127, 127]  # +z -> diamond centre
    assert packed[0, 1, :2].tolist() == [255, 127]  # +x -> u max
    assert packed[0, 2, :2].tolist() == [127, 255]  # +y -> v max


def test_unpack_octahedral_decodes_hand_built_bytes():
    """Hand-built octahedral bytes decode to their hand-derived normals.

    Built from literals rather than by round-tripping through ``pack``, so it
    is the unpack side's own channel order and rescale under test. Breaks if
    the U/V channels are read in the wrong order — swapping them is invisible
    to a round-trip, which just applies the same swap twice.
    """
    # u = 255 -> +1, v = 127 -> ~0, so the first pixel is the +x normal. The
    # depth channel is a valid mid-range level, not the invalid 0.
    packed = np.array(
        [[[255, 127, 128], [127, 255, 128], [127, 127, 128]]], dtype=np.uint8
    )

    normals, _ = unpack_octahedral_depth(packed)

    assert np.allclose(normals[0, 0], [1, 0, 0], atol=0.05)  # u max -> +x
    assert np.allclose(normals[0, 1], [0, 1, 0], atol=0.05)  # v max -> +y
    assert np.allclose(normals[0, 2], [0, 0, 1], atol=0.05)  # centre -> +z


def test_pack_octahedral_round_trips_back_facing_normals():
    """Normals pointing away from the camera survive pack -> unpack.

    Breaks if the nz < 0 unfolding scales by ``np.sign`` of the raw
    coordinate: sign(0) is 0, so a coordinate that is exactly 0 zeroes the
    other unfolded coordinate and the normal comes back rotated by up to 90
    degrees.
    """
    normals = _unit([[[0, 0, -1], [0.001, 0, -0.999], [0, 0.001, -0.999]]])

    decoded, _ = unpack_octahedral_depth(_pack(normals))

    assert np.allclose(decoded, normals, atol=0.05)


def test_pack_octahedral_round_trips_random_unit_normals():
    """Both hemispheres survive pack -> unpack within 8-bit octahedral error."""
    rng = np.random.default_rng(0)
    normals = _unit(rng.normal(size=(32, 48, 3)))

    decoded, _ = unpack_octahedral_depth(_pack(normals))

    assert np.allclose(decoded, normals, atol=0.05)


def test_pack_octahedral_invalid_pixels_are_zero():
    """Pixels outside the model mask pack to (0, 0, 0) in every channel.

    Breaks if the mask is ignored and the model's zeroed normals / infinite
    depth are encoded as real values.
    """
    normals = np.zeros((1, 2, 3), dtype=np.float32)
    normals[0, 0] = [0, 0, 1]
    valid = np.array([[True, False]])
    depth = np.array([[MID_DEPTH, np.inf]], dtype=np.float32)

    packed = _pack(normals, depth, valid)

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_pack_octahedral_infinite_depth_reads_back_invalid():
    """+inf depth (MoGe's masked-out sentinel) packs to 0, not to the far clip.

    The log codec treats +inf as a *valid* measurement and clips it to the
    range maximum, so an unguarded encode would make every masked-out pixel
    read back as the farthest depth in the scene.
    """
    normals = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    depth = np.array([[MID_DEPTH, np.inf]], dtype=np.float32)

    packed = _pack(normals, depth)

    assert packed[0, 1, 2] == 0
    _, decoded_depth = unpack_octahedral_depth(packed)
    assert decoded_depth[0, 1] == 0.0


def test_pack_octahedral_masks_pixels_whose_depth_alone_looks_valid():
    """A ``valid=False`` pixel packs to (0, 0, 0) even with in-range depth.

    Breaks if the ``valid`` argument is ignored. The depth here is finite and
    mid-range on both pixels, so the packer's own finiteness guard cannot
    catch it — the mask is the only thing that can.
    """
    normals = np.array([[[0, 0, 1], [0, 0, 1]]], dtype=np.float32)
    depth = np.full((1, 2), MID_DEPTH, dtype=np.float32)

    packed = _pack(normals, depth, np.array([[True, False]]))

    assert packed[0, 0].tolist() != [0, 0, 0]
    assert packed[0, 1].tolist() == [0, 0, 0]


def test_pack_octahedral_accepts_uint16_millimetres():
    """A uint16 depth in millimetres is read as mm, not as metres.

    The log codec auto-detects uint16-mm input, and packing must preserve the
    dtype for that to survive. Breaks if a float cast creeps in ahead of the
    codec: 500 mm would then be read as 500 m and clip to the far range end,
    decoding back as 1.25 m instead of 0.5 m.
    """
    normals = np.array([[[0, 0, 1]]], dtype=np.float32)

    _, decoded_depth = unpack_octahedral_depth(
        _pack(normals, np.array([[500]], dtype=np.uint16))
    )

    assert np.allclose(decoded_depth, 0.5, rtol=DEPTH_RTOL)


def test_pack_octahedral_depth_channel_round_trips_metres():
    """The depth channel round-trips metric metres within codec quantization."""
    depths = np.array([[0.3, 0.5, 1.0, 1.2]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (1, 4, 1))

    _, decoded_depth = unpack_octahedral_depth(_pack(normals, depths))

    assert np.allclose(decoded_depth, depths, rtol=DEPTH_RTOL)


def test_pack_octahedral_depth_channel_is_log_not_linear():
    """The geometric mid-range depth sits mid-scale: the codec is logarithmic.

    Breaks if the depth channel is swapped for a plain linear
    (d - min) / (max - min) normalization, which would put this depth at 79.
    """
    # sqrt(min_depth * max_depth) with the codec's 0.001 m log shift.
    geometric_mid = 0.5604
    normals = np.array([[[0, 0, 1]]], dtype=np.float32)

    packed = _pack(normals, np.array([[geometric_mid]], dtype=np.float32))

    assert 127 <= packed[0, 0, 2] <= 129
