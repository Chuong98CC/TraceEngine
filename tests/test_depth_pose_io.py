"""Round-trip tests for the indexed depth_pose store (CPU only)."""
import struct

import numpy as np
import pytest

from utils.depth_pose_io import (
    DEPTH_FILE,
    POSES_FILE,
    DepthContainerReader,
    DepthContainerWriter,
    DepthPoseReader,
    DepthPoseWriter,
)


def _depth(h, w, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w), dtype=np.float32) * 1.0 + 0.3).astype(np.float32)


def test_container_roundtrip_bit_identical(tmp_path):
    h, w, t = 8, 12, 5
    frames = [_depth(h, w, i) for i in range(t)]
    path = tmp_path / DEPTH_FILE
    with DepthContainerWriter(path, h, w) as writer:
        for f in frames:
            writer.append(f)

    with DepthContainerReader(path) as reader:
        assert reader.frame_count == t
        assert reader.shape == (h, w)
        for i, expected in enumerate(frames):
            got = reader.read(i)
            assert got.dtype == np.float32
            assert got.shape == (h, w)
            # the codec is lossy at 8 bits, so the contract is: reading a
            # container reproduces the codec's own encode/decode round-trip
            # of the frame that was appended
            assert np.array_equal(got, expected_quantized(expected))


def expected_quantized(depth_m):
    from utils.depth_utils import LogDepthToUint8Transform

    codec = LogDepthToUint8Transform()
    return codec.decode(codec.encode(depth_m))


def test_container_random_access_matches_sequential(tmp_path):
    h, w, t = 6, 6, 7
    frames = [_depth(h, w, i) for i in range(t)]
    path = tmp_path / DEPTH_FILE
    with DepthContainerWriter(path, h, w) as writer:
        for f in frames:
            writer.append(f)

    with DepthContainerReader(path) as reader:
        sequential = reader.read_range(0, t)
        for i in (0, 3, t - 1):
            assert np.array_equal(reader.read(i), sequential[i])


def test_unfinalized_container_is_scannable(tmp_path):
    """A writer killed before close() leaves header + records and no index:
    the reader recovers every frame by scanning the record framing."""
    h, w, t = 5, 5, 3
    frames = [_depth(h, w, i) for i in range(t)]
    path = tmp_path / DEPTH_FILE
    writer = DepthContainerWriter(path, h, w)
    for f in frames:
        writer.append(f)
    writer._fh.flush()
    writer._fh.close()  # abandon: no index, no trailer, no header patch

    with DepthContainerReader(path) as reader:
        assert reader.frame_count == t
        for i, expected in enumerate(frames):
            assert np.array_equal(reader.read(i), expected_quantized(expected))


def test_container_ignores_a_torn_tail(tmp_path):
    """A run killed mid-append leaves a partial last record: the complete
    prefix is recovered and the torn tail is ignored."""
    h, w, t = 5, 5, 4
    frames = [_depth(h, w, i) for i in range(t)]
    path = tmp_path / DEPTH_FILE
    with DepthContainerWriter(path, h, w) as writer:
        for f in frames:
            writer.append(f)
    healthy = path.read_bytes()

    # chop the trailer (16 B) plus a byte of the index: the records are all
    # intact, so the short index must fall back to scanning and recover them
    scan_path = tmp_path / "scan.lz4"
    scan_path.write_bytes(healthy[: len(healthy) - 17])
    with DepthContainerReader(scan_path) as reader:
        assert reader.frame_count == t
        assert np.array_equal(reader.read(t - 1), expected_quantized(frames[t - 1]))

    # chop into the last record's payload: only the complete prefix survives.
    # The tail is the index ((t + 1) uint64 offsets) plus the 16-byte trailer,
    # so cut 4 bytes past it to land inside the last record.
    cut = (t + 1) * 8 + 16 + 4
    torn_path = tmp_path / "torn.lz4"
    torn_path.write_bytes(healthy[: len(healthy) - cut])
    with DepthContainerReader(torn_path) as reader:
        assert 1 <= reader.frame_count < t
        assert np.array_equal(reader.read(0), expected_quantized(frames[0]))


def test_zero_frame_container(tmp_path):
    path = tmp_path / DEPTH_FILE
    with DepthContainerWriter(path, 4, 4):
        pass
    with DepthContainerReader(path) as reader:
        assert reader.frame_count == 0


def test_poses_roundtrip(tmp_path):
    t = 4
    indices = np.array([10, 14, 18, 22], dtype=np.int64)
    ext = np.arange(t * 12, dtype=np.float32).reshape(t, 3, 4)
    intr = np.arange(t * 9, dtype=np.float32).reshape(t, 3, 3)
    with DepthPoseWriter(tmp_path, 8, 12) as writer:
        for i in range(t):
            writer.add(int(indices[i]), _depth(8, 12, i), ext[i], intr[i])

    reader = DepthPoseReader(tmp_path)
    assert np.array_equal(reader.frame_indices, indices)
    assert reader.shape == (8, 12)
    assert reader.index_of(18) == 2
    assert np.array_equal(reader.pose_at(18)[0], ext[2])
    assert np.array_equal(reader.pose_at(18)[1], intr[2])
    assert np.array_equal(reader.depth_at(18), expected_quantized(_depth(8, 12, 2)))
    assert np.array_equal(reader.depth_range(0, t), reader.depth_range(0, None))
    assert (tmp_path / POSES_FILE).is_file()
    assert DepthPoseReader.is_complete(tmp_path)

    with pytest.raises(KeyError):
        reader.pose_at(11)


def test_is_complete_needs_poses(tmp_path):
    """A pass-1 container with no poses.npz is an incomplete segment — the
    check must not need a reader (constructing one loads poses.npz)."""
    with DepthContainerWriter(tmp_path / DEPTH_FILE, 4, 4) as writer:
        writer.append(_depth(4, 4, 0))
    assert not DepthPoseReader.is_complete(tmp_path)
    with DepthPoseWriter(tmp_path, 4, 4) as writer:
        writer.add(7, _depth(4, 4, 0), np.eye(3, 4, dtype=np.float32),
                   np.eye(3, dtype=np.float32))
    assert DepthPoseReader.is_complete(tmp_path)
