"""Golden check: piling the legacy per-frame depth_pose pair into a container
must reproduce it bit-for-bit. Skipped when the corpus is not on disk."""
import os
from pathlib import Path

import numpy as np
import pytest

from utils.depth_pose_io import DepthPoseReader, DepthPoseWriter
from utils.depth_utils import load_depth_lz4

_CANDIDATES = (
    os.environ.get("ASTRIBOT_DATA_ROOT"),
    "/data/astribot_making_coffee_vlva_full",
    str(Path(__file__).resolve().parents[1]
        / "astribot_making_coffee_vlva_full"),
)


def _legacy_segment():
    """First legacy depth_pose camera dir with per-frame pairs, or None."""
    for root in _CANDIDATES:
        if not root:
            continue
        legacy = Path(root) / "eps_data" / "depth_pose"
        if not legacy.is_dir():
            continue
        for npz in sorted(legacy.glob("ep*/subtask_*/depth_*/*.npz")):
            return npz.parent
    return None


SEGMENT = _legacy_segment()


def test_container_matches_save_depth_lz4(tmp_path):
    """The drop-in guarantee, exact: appending a float depth map to a
    container and reading it back is bit-identical to the old
    save_depth_lz4 -> load_depth_lz4 round-trip of the same values."""
    from utils.depth_utils import save_depth_lz4

    h, w = 8, 12
    rng = np.random.default_rng(0)
    frames = [
        (rng.random((h, w), dtype=np.float32) * 0.9 + 0.3).astype(np.float32)
        for _ in range(4)
    ]
    for i, depth_m in enumerate(frames):
        legacy_path = tmp_path / f"frame_{i:06d}.lz4"
        save_depth_lz4(depth_m, legacy_path)
        legacy = load_depth_lz4(legacy_path, (h, w))

        with DepthPoseWriter(tmp_path / f"c{i}", h, w) as writer:
            writer.add(i, depth_m, np.eye(3, 4, dtype=np.float32),
                       np.eye(3, dtype=np.float32))
        with DepthPoseReader(tmp_path / f"c{i}") as reader:
            assert np.array_equal(reader.depth(0), legacy), f"frame {i} differs"


@pytest.mark.skipif(SEGMENT is None,
                    reason="no legacy eps_data/depth_pose corpus on disk")
def test_legacy_segment_converts(tmp_path):
    """A legacy segment loads through the container: frame order and poses
    exact, depth within one quantization level.

    Converting is lossier than regenerating: a legacy file holds
    encode(depth_m), so appending its decode re-quantizes an already
    quantized map and can shift a pixel by one of the 255 levels (measured
    0.63% relative on this corpus). The pipeline itself never does this —
    it appends the float depth it computed (see
    test_container_matches_save_depth_lz4 above)."""
    stems = sorted(p.stem for p in SEGMENT.glob("*.npz"))
    assert stems, f"no per-frame npz under {SEGMENT}"

    poses = [np.load(SEGMENT / f"{s}.npz") for s in stems]
    legacy_depth = [
        load_depth_lz4(SEGMENT / f"{s}.lz4", tuple(int(v) for v in p["shape"]))
        for s, p in zip(stems, poses)
    ]
    height, width = legacy_depth[0].shape

    with DepthPoseWriter(tmp_path, height, width) as writer:
        for stem, depth, pose in zip(stems, legacy_depth, poses):
            writer.add(int(stem.rsplit("_", 1)[-1]), depth,
                       pose["extrinsics"], pose["intrinsics"])

    with DepthPoseReader(tmp_path) as reader:
        assert np.array_equal(
            reader.frame_indices,
            np.array([int(s.rsplit("_", 1)[-1]) for s in stems], dtype=np.int64),
        )
        assert reader.shape == (height, width)
        for i, expected in enumerate(legacy_depth):
            got = reader.depth(i)
            rel = np.abs(got - expected) / np.maximum(expected, 1e-6)
            assert rel.max() <= 0.01, f"frame {i} drifted {rel.max():.4f}"
        for i, pose in enumerate(poses):
            ext, intr = reader.pose_at(int(reader.frame_indices[i]))
            assert np.array_equal(ext, pose["extrinsics"].astype(np.float32))
            assert np.array_equal(intr, pose["intrinsics"].astype(np.float32))
