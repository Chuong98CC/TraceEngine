"""The Step-2 finalize pass: scaling + pose write, no model needed."""
from pathlib import Path

import numpy as np
import pytest

from depth_models.streaming.base_streaming import _finalize_depth_pose
from utils.depth_pose_io import (
    DEPTH_FILE,
    DepthContainerWriter,
    DepthPoseReader,
)

H, W = 6, 8


def test_finalize_applies_scale_and_writes_poses(tmp_path):
    out_dirs = [str(tmp_path / "cam_head"), str(tmp_path / "cam_torso")]
    # pass 1: each camera's container holds one raw (unscaled) frame
    for d in out_dirs:
        with DepthContainerWriter(Path(d) / DEPTH_FILE, H, W) as writer:
            writer.append(np.full((H, W), 0.5, np.float32))

    # one step, belonging to chunk 1, two cameras; chunk 1 -> chunk 0 scale 2
    # (both poses are per-view stacked, as BaseStreaming.run() records them)
    frame_meta = [(100, 1, np.tile(np.eye(3, dtype=np.float32) * 2, (2, 1, 1)))]
    all_extrinsics = [np.tile(np.eye(3, 4, dtype=np.float32), (2, 1, 1))]
    sim3_cum = [(2.0, np.eye(3, dtype=np.float32), np.zeros(3, np.float32))]

    _finalize_depth_pose(out_dirs, frame_meta, all_extrinsics, sim3_cum,
                         (H, W), slots=[0, 1])

    for d in out_dirs:
        with DepthPoseReader(d) as reader:
            assert reader.frame_indices.tolist() == [100]
            assert reader.shape == (H, W)
            assert reader.depth(0).mean() > 0.9  # 0.5 m scaled by s=2
            ext, intr = reader.pose_at(100)
            assert ext.shape == (3, 4)
            assert ext[0, 0] == pytest.approx(0.5)  # eye @ inv(S), S = 2*eye
            assert intr[0, 0] == pytest.approx(2.0)
        assert not (Path(d) / (DEPTH_FILE + ".tmp")).exists()
