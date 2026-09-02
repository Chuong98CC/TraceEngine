import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode

from utils.image_io import (
    imagenet_normalize,
    letterbox,
    to_image_tensor,
    to_pixel_uint8,
)


def _rand_chw(seed=0, h=360, w=640):
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.integers(0, 256, (h, w, 3), dtype=np.uint8)).permute(2, 0, 1)


def _cv2_letterbox(img_hwc, target_h, target_w, scale_mode="trunc2"):
    """Legacy reference: cv2 uint8 pipeline + the shared float64 scale math."""
    import cv2

    oh, ow = img_hwc.shape[:2]
    raw = min(target_w / ow, target_h / oh)
    if scale_mode == "trunc2":
        scale = np.floor(raw * 100.0) / 100.0
        if scale <= 0:
            scale = raw
        nw, nh = int(ow * scale), int(oh * scale)
    else:
        nw, nh = round(ow * raw), round(oh * raw)
    resized = cv2.resize(img_hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pt, pl = (target_h - nh) // 2, (target_w - nw) // 2
    padded = cv2.copyMakeBorder(resized, pt, target_h - nh - pt, pl,
                                target_w - nw - pl, cv2.BORDER_CONSTANT, value=0)
    return padded, nh, nw, pt, pl


def test_to_image_tensor_path(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(_rand_chw(1).permute(1, 2, 0).numpy()).save(p)
    out = to_image_tensor(str(p))
    assert out.dtype == torch.uint8 and out.ndim == 3 and out.shape[0] == 3


def test_to_image_tensor_numpy_matches_manual():
    hwc = _rand_chw(2).permute(1, 2, 0).numpy()
    out = to_image_tensor(hwc)
    manual = torch.from_numpy(np.ascontiguousarray(hwc)).permute(2, 0, 1)
    assert torch.equal(out, manual)


def test_to_image_tensor_nonwritable_numpy():
    hwc = _rand_chw(3).permute(1, 2, 0).numpy()
    hwc.setflags(write=False)
    assert to_image_tensor(hwc).dtype == torch.uint8


def test_to_image_tensor_errors():
    with pytest.raises(ValueError):
        to_image_tensor(torch.zeros(4, 4))            # not CHW
    with pytest.raises(TypeError):
        to_image_tensor(123)                           # unsupported type


def test_to_pixel_uint8_passthrough_and_rescale():
    u = torch.zeros(3, 4, 4, dtype=torch.uint8)
    assert to_pixel_uint8(u) is u
    f = torch.tensor([0.0, 0.5, 1.0]).view(3, 1, 1).expand(3, 4, 4).clone()
    out = to_pixel_uint8(f)
    assert out.dtype == torch.uint8
    assert out[0, 0, 0] == 0 and out[1, 0, 0] == 128 and out[2, 0, 0] == 255


def test_letterbox_meta_trunc2_exact():
    x = _rand_chw(h=600, w=1000)
    padded, meta = letterbox(x, 480, 640)   # scale min(640/1000, 480/600) = 0.64
    assert meta == {"orig_h": 600, "orig_w": 1000, "scale_factor": 0.64,
                    "tile_h": 384, "tile_w": 640, "pad_top": 48, "pad_left": 0}
    assert padded.shape == (3, 480, 640) and padded.dtype == torch.uint8
    assert int(padded[:, :48, :].max()) == 0 and int(padded[:, -48:, :].max()) == 0


def test_letterbox_matches_cv2_reference():
    for scale_mode in ("trunc2", "round"):
        x = _rand_chw(seed=7)
        hwc = x.permute(1, 2, 0).numpy()
        padded, meta = letterbox(x, 480, 640, scale_mode=scale_mode)
        ref, _, _, _, _ = _cv2_letterbox(hwc, 480, 640, scale_mode)
        ref_t = torch.from_numpy(ref).permute(2, 0, 1)
        assert padded.shape == ref_t.shape
        diff = (padded.float() - ref_t.float()).abs()
        assert diff.max() <= 3 and (diff <= 1).float().mean() >= 0.999


def test_letterbox_round_mode_geometry():
    x = _rand_chw(h=600, w=1000)
    padded, meta = letterbox(x, 480, 640, scale_mode="round")
    # raw = min(640/1000, 480/600) = 0.64 exactly; round(1000*0.64)=640 etc.
    assert (meta["tile_h"], meta["tile_w"]) == (384, 640)
    assert meta["scale_factor"] == 0.64


def test_letterbox_float_entry_and_channel_check():
    x = _rand_chw()
    f = x.float() / 255.0
    padded, meta = letterbox(f, 480, 640)
    assert padded.dtype == torch.uint8 and meta["orig_h"] == 360
    with pytest.raises(ValueError):
        letterbox(torch.zeros(1, 10, 10), 480, 640)  # not 3 channels


def test_imagenet_normalize_known_values():
    x = torch.tensor([[[0]], [[128]], [[255]]], dtype=torch.uint8)  # (3,1,1)
    out = imagenet_normalize(x)
    exp = torch.tensor([(0 / 255 - 0.485) / 0.229,
                        (128 / 255 - 0.456) / 0.224,
                        (255 / 255 - 0.406) / 0.225], dtype=torch.float32).view(3, 1, 1)
    assert torch.allclose(out, exp, atol=1e-6)
    assert out.dtype == torch.float32
