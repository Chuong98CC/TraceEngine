"""CPU-only unit tests for utils.depth_utils unprojection helpers."""

import numpy as np

from utils.depth_utils import unproject_depth_map_to_point_map


def test_unproject_channel_first_depth_layout():
    """4D depth must be read channel-first (N, 1, H, W), not channel-last.

    Regression: the old code sliced ``depth_map[..., 0]`` (the HWC
    convention), which on a channel-first (N, 1, H, W) batch takes the last
    dim (W) and yields a garbage (N, 1, H) "depth" — every axis misread.
    """
    n, h, w = 2, 4, 6
    fx = fy = 100.0
    cx, cy = w / 2, h / 2
    depth = np.arange(1.0, 1.0 + n * h * w, dtype=np.float32).reshape(n, 1, h, w)
    intrinsic = np.repeat(
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)[None], n, axis=0
    )
    extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)

    points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    assert points.shape == (n, h, w, 3)

    # world = camera (identity extrinsics): X = (x - cx)/fx * d, Y = (y - cy)/fy * d
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = depth[:, 0]
    expected = np.stack(
        [(x[None] - cx) / fx * d, (y[None] - cy) / fy * d, d], axis=-1
    )
    np.testing.assert_allclose(points, expected, atol=1e-5)

    # (N, H, W) input gives the same result as the channel-first (N, 1, H, W)
    flat = unproject_depth_map_to_point_map(depth[:, 0], extrinsic, intrinsic)
    np.testing.assert_allclose(points, flat, atol=1e-6)
