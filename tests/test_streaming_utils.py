"""CPU-only unit tests for utils.streaming_utils batch resize helpers."""

import numpy as np
import torch

from utils.streaming_utils import resize_batch_to_inference


def test_resize_batch_to_inference_intrinsics_scaling():
    """Intrinsics scale factors must come from the CHW frame dims.

    Regression: when the frame loaders moved to CHW (T, 3, H0, W0), the
    scale factors were still read from ``video.shape[1:3]`` — the channel
    dim (3) became ``orig_h`` and the height became ``orig_w``, corrupting
    fy/cx/cy by (inf_h - 1) / 2 (~239x at 480p) and skewing fx.
    """
    t, h0, w0 = 4, 60, 80
    inf_h, inf_w = 30, 40
    video = torch.zeros((t, 3, h0, w0), dtype=torch.uint8)
    geo = {
        "depth": np.zeros((t, h0, w0), dtype=np.float32),
        "intrs": np.repeat(np.eye(3, dtype=np.float32)[None], t, axis=0),
        "extrs": np.repeat(np.eye(4, dtype=np.float32)[None], t, axis=0),
    }
    _, depths, intrs, _ = resize_batch_to_inference(video, geo, inf_h, inf_w)

    sy = (inf_h - 1) / (h0 - 1)
    sx = (inf_w - 1) / (w0 - 1)
    np.testing.assert_allclose(
        intrs[0].numpy(),
        np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float32),
        atol=1e-6,
    )
    assert depths.shape == (t, inf_h, inf_w)
