"""The shared readers against a synthetic depth_pose folder (CPU only)."""
from pathlib import Path

import numpy as np
import pytest

from utils.depth_pose_io import DepthPoseWriter
from utils.streaming_utils import (
    compute_global_depth_roi,
    load_npz_batch,
    load_pair,
    load_stream_data,
)

H, W = 6, 8
INDICES = [100, 104, 108]


def _pose_dir(tmp_path) -> Path:
    with DepthPoseWriter(tmp_path, H, W) as writer:
        for i, idx in enumerate(INDICES):
            writer.add(idx, np.full((H, W), 0.5 + 0.1 * i, np.float32),
                       np.eye(3, 4, dtype=np.float32),
                       np.eye(3, dtype=np.float32) * (i + 1))
    return tmp_path


def test_load_stream_data_by_frame_index(tmp_path):
    d = _pose_dir(tmp_path)
    depth, ext, intr = load_stream_data(d, INDICES[1])
    assert depth.shape == (H, W) and depth.dtype == np.float32
    assert ext.shape == (3, 4)
    assert intr[0, 0] == pytest.approx(2.0)
    with pytest.raises(KeyError):
        load_stream_data(d, 101)


def test_load_npz_batch_slice(tmp_path):
    d = _pose_dir(tmp_path)
    geo = load_npz_batch(d, INDICES, 1, 3)
    assert geo["depth"].shape == (2, H, W)
    assert geo["extrs"].shape == (2, 4, 4)  # padded to homogeneous 4x4
    assert geo["intrs"].shape == (2, 3, 3)
    assert geo["intrs"][1, 0, 0] == pytest.approx(3.0)


def test_load_pair_rejects_mismatched_camera_lists():
    """Parallel lists: a length mismatch must not silently drop a camera."""
    with pytest.raises(ValueError):
        load_pair(INDICES[0], image_dirs=["a", "b"], depth_dirs=["a"])


def test_compute_global_depth_roi(tmp_path):
    d = _pose_dir(tmp_path)
    roi = compute_global_depth_roi(d, INDICES, H, W)
    assert roi.shape == (2,)
    assert float(roi[1]) > float(roi[0])
