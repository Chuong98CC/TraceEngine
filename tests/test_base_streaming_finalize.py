"""The Step-2 finalize pass: scaling + pose write, no model needed."""
import gc
from pathlib import Path

import numpy as np
import pytest

from depth_models.streaming.base_streaming import BaseStreaming, _finalize_depth_pose
from utils.depth_pose_io import (
    DEPTH_FILE,
    POSES_FILE,
    DepthContainerReader,
    DepthContainerWriter,
    DepthPoseReader,
    write_poses,
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


def test_run_drops_the_stale_poses_marker(tmp_path):
    """run() must clear poses.npz before pass 1 truncates the container.

    poses.npz is the completion marker: a leftover one from an earlier run
    would make a segment that dies in this pass (container already truncated,
    no poses written) look finished to DepthPoseReader.is_complete().
    """
    in_dir = tmp_path / "in" / "cam_head"
    in_dir.mkdir(parents=True)
    for t in range(2):
        (in_dir / f"frame_{t:06d}.jpg").touch()

    out_dir = tmp_path / "depth_pose" / "cam_head"
    out_dir.mkdir(parents=True)
    # an earlier run's finished folder: container + completion marker
    with DepthContainerWriter(out_dir / DEPTH_FILE, H, W) as writer:
        writer.append(np.full((H, W), 0.5, np.float32))
    write_poses(out_dir, np.asarray([999], np.int64),
                np.zeros((1, 3, 4), np.float32),
                np.zeros((1, 3, 3), np.float32), (H, W))
    assert DepthPoseReader.is_complete(out_dir)

    class _DiesInPassOne(BaseStreaming):
        """One camera, one image per chunk; dies on the second chunk."""

        calls = 0

        def _process_chunk(self, start, end):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("died inside pass 1")
            return {
                "depth": np.full((1, H, W), 1.0, np.float32),
                "conf": np.ones((1, H, W), np.float32),
                "extrinsics": np.eye(3, 4, dtype=np.float32)[None],
                "intrinsics": np.eye(3, dtype=np.float32)[None],
            }

    stream = _DiesInPassOne(input_dirs=[str(in_dir)],
                            save_dir=str(tmp_path / "depth_pose"),
                            config={}, chunk_size=1)
    with pytest.raises(RuntimeError, match="died inside pass 1"):
        stream.run()

    assert not (out_dir / POSES_FILE).exists(), "stale completion marker"
    assert not DepthPoseReader.is_complete(out_dir)

    # ...and the reason it mattered: pass 1 already replaced the container
    # (its writer died mid-pass, so the reader scans the record framing).
    del stream
    gc.collect()
    with DepthContainerReader(out_dir / DEPTH_FILE) as reader:
        assert reader.frame_count == 1
        assert reader.read(0).mean() == pytest.approx(1.0, abs=0.01)  # not 0.5
