# Unified Image Input Handling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route every torch-based model runtime's image read + geometry through the shared RGB tensor-space helpers in `src/utils/image_io.py` (`to_image_tensor` → `letterbox`/`imagenet_normalize`), with a uniform decode-only `_load_image` member, feeding `torch.Tensor`s straight to the exported graphs.

**Architecture:** image_io gains three pure-torch helpers (`to_pixel_uint8`, `letterbox` with per-model geometry knobs, `imagenet_normalize`). Six runtimes (DA3, VGGT-Omega, RoMaV2, SAM3, Any2Full, TAPIP3D loaders) adopt them; the online streamer stores RGB; **WAFT is untouched**. A committed golden harness (capture → compare) gates every runtime rewrite, since the repo has no tests.

**Tech Stack:** PyTorch / torchvision (v2 transforms), numpy, Python ≥3.12.1 <3.13, uv (`--group dev` already contains pytest). cv2 remains only for depth/masks/video/viz and WAFT.

**Spec:** `docs/superpowers/specs/2026-09-02-unified-image-input-design.md`

## Global Constraints

- Channel contract: numpy/tensor images are **RGB uint8**. Float `[0,1]` CHW *tensor* sources are rescaled once at entry (`to_pixel_uint8`). Float numpy is dropped (RoMaV2-only feature).
- WAFT (`base_waft.py`, `waft.py`, `infer_waft.py`) and `MonoDepthTRT`/`StereoDepthTRT` are **never touched** in this plan. WAFT keeps `bgr_input`; call sites that still feed it BGR must flip there.
- Tensor-first: torch.export graphs receive tensors; numpy remains only for model outputs, depth/masks, and WAFT feeds.
- `letterbox` geometry modes: `"trunc2"` (DA3; WAFT's convention, floor×100/100, `int()`) and `"round"` (VGGT, `round()`); meta keys `orig_h/orig_w/scale_factor/tile_h/tile_w/pad_top/pad_left` must be **exact** vs the legacy cv2/PIL math (float64).
- Resize uses `torchvision.transforms.v2.functional.resize` on uint8 with `antialias=False` (preserves integer quantization).
- Golden compare gate: after each runtime rewrite, run the compare for that runtime and require PASS before committing.
- Run unit tests with `uv run --group dev pytest tests/ -q` (pytest already in `[dependency-groups] dev` — no pyproject change).
- Goldens live under `output/verify_image_inputs/` (git-ignored — `output` is already in .gitignore). Never alter the capture parameters between capture and compare.
- One commit per task; commit message body ends with `Co-Authored-By: Claude <noreply@anthropic.com>`.

---

### Task 1: Golden capture/compare harness + baseline capture

**Files:**
- Create: `tools/general_test/module/compare_image_inputs.py`
- Create: `scripts/general_test/verify_image_inputs.sh`

**Interfaces:**
- Produces: CLI `compare_image_inputs.py --mode capture|compare [--runtime da3|vggt|romav2|sam3|any2full|tapip3d|all] [--golden-dir DIR] [--frames-dir DIR] [--e2e ...]`. Capture writes `<golden-dir>/<runtime>.npz`; compare re-runs the same probe calls, diffs against the npz, prints per-entry `max|p99` and a PASS/FAIL verdict, exit code 0 iff all entries pass.

- [ ] **Step 1: Write the harness**

`tools/general_test/module/compare_image_inputs.py`:

```python
# -*- coding: utf-8 -*-
"""Golden capture/compare for runtime image-preprocess refactors.

The preprocess code paths change shape across the refactor (numpy<->tensor,
deleted module helpers), so probes call only *stable public entry points*
that exist before and after: DA3 preprocess_views, VGGT BaseVGGTOmega.infer
via a recording probe subclass, RoMaV2 match_pair, SAM3 preprocess_image,
Any2Full preprocess, streaming_utils.load_batch_frames.

When the da3/vggt weights are present under weights/, capture and compare
also run real end-to-end model forwards (saved as e2e.npz, mode=report —
printed maxima, never asserted).

Usage:
  # before the refactor (on current code):
  python tools/general_test/module/compare_image_inputs.py capture
  # after each runtime rewrite:
  python tools/general_test/module/compare_image_inputs.py compare --runtime da3
Wrappers: scripts/general_test/verify_image_inputs.sh
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "astribot_test_imgs"
GOLDEN_DEFAULT = Path("output/verify_image_inputs")
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "weights"

# Entries are (key, value) with key tagged "<name>:<mode>",
# mode in {"exact", "lsb", "fp32"}: exact = bit-identical (meta/crops),
# lsb = uint8 pixel arrays (p99 <= 1, max <= 3), fp32 = normalized arrays
# (max <= 2e-2).  e2e outputs use mode "report" (printed, not asserted).


def _np(x) -> np.ndarray:
    """Normalize tensor/numpy outputs to CPU numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def _frames(frames_dir: Path, n: int = 2) -> list[Path]:
    imgs = sorted(p for p in frames_dir.rglob("*.jpg") if "depth" not in p.parts)
    if len(imgs) < n:
        raise SystemExit(f"Need >= {n} jpgs under {frames_dir}, found {len(imgs)}")
    return imgs[:n]


def _align(a: np.ndarray) -> np.ndarray:
    """HWC-numpy -> CHW so pre/post refactor arrays compare regardless of layout.

    Handles (H,W,3) -> (3,H,W), (T,H,W,3) -> (T,3,H,W); CHW/NCHW pass through.
    """
    if a.ndim == 3 and a.shape[-1] == 3 and a.shape[0] != 3:
        return a.transpose(2, 0, 1)
    if a.ndim == 4 and a.shape[-1] == 3 and a.shape[1] != 3:
        return a.transpose(0, 3, 1, 2)
    return a


def _first_pt2(d: str, pattern: str = "*.pt2") -> str | None:
    hits = sorted(Path(d).glob(pattern))
    return str(hits[0]) if hits else None


# ---------------------------------------------------------------------------
# Probes: one per runtime, calling only APIs that exist on both sides
# ---------------------------------------------------------------------------

def _probe_da3(frames_dir: Path):
    """BaseDA3Model.preprocess_views needs only target_h/w from a subclass."""
    import torch
    from depth_models.da3.model.base_da3 import BaseDA3Model

    class _P(BaseDA3Model):
        pass

    probe = _P()
    probe.target_h, probe.target_w = 490, 644  # any-view-style geometry
    imgs = [str(p) for p in _frames(frames_dir)]
    batch, metas = probe.preprocess_views(imgs)
    items = [("da3_batch:fp32", _np(batch))]
    for i, m in enumerate(metas):
        for k, v in m.items():
            items.append((f"da3_meta{i}_{k}:exact", np.asarray(v)))
    return items


def _probe_vggt(frames_dir: Path):
    """BaseVGGTOmega.infer with a recording _forward that fabricates outputs
    shaped like the real ones (postprocess still runs on them)."""
    import numpy as np
    import torch
    from depth_models.vggt_omega.base_vggt_omega import BaseVGGTOmega

    class _P(BaseVGGTOmega):
        def __init__(self):
            self.target_h, self.target_w = 518, 518
            self.inputs = [{"name": "images"}]
            self.recorded = None

        def _forward(self, feed):
            self.recorded = {k: _np(v) for k, v in feed.items()}
            batch = _np(next(iter(feed.values())))
            n, h, w = batch.shape[1], batch.shape[3], batch.shape[4]
            pose = np.zeros((1, n, 9), dtype=np.float32)
            pose[..., 6] = 1.0        # identity quat (scalar last)
            pose[..., 7:9] = np.pi / 3  # sane fov_h/fov_w so intrinsics stay finite
            return {
                "pose_enc": pose,
                "depth": np.zeros((1, n, h, w), np.float32),
                "depth_conf": np.zeros((1, n, h, w), np.float32),
            }

    probe = _P()
    out = probe.infer([str(p) for p in _frames(frames_dir)])
    items = [(f"vggt_feed_{k}:fp32", v) for k, v in probe.recorded.items()]
    for k, v in out.items():
        mode = "exact" if k == "crop" else "fp32"
        items.append((f"vggt_out_{k}:{mode}", np.asarray(v)))
    return items


def _probe_romav2(frames_dir: Path, device="cuda"):
    """match_pair end-to-end (module load on cuda); warp outputs deterministic."""
    from det_seg_models.romav2.romav2 import RoMaV2PT2

    model = RoMaV2PT2(device=device)
    a, b = (str(p) for p in _frames(frames_dir))
    preds = model.match_pair(a, b)
    items = []
    for k, v in preds.items():
        if v is not None:
            items.append((f"romav2_{k}:fp32", _np(v)))
    return items


def _probe_sam3(frames_dir: Path):
    from det_seg_models.sam3.sam3_model import Sam3Image

    ckpt = _first_pt2(str(DEFAULT_WEIGHTS / "sam3"))
    if ckpt is None:
        raise SystemExit("No SAM3 .pt2 under weights/sam3")
    model = Sam3Image(ckpt)
    t = model.preprocess_image(str(_frames(frames_dir, 1)[0]), device="cpu")
    return [("sam3_preprocess:fp32", _np(t))]


def _probe_any2full(frames_dir: Path):
    from PIL import Image

    from depth_models.a3f.any2full import Any2Full_PT2

    ckpt = _first_pt2(str(DEFAULT_WEIGHTS / "any2full"))
    if ckpt is None:
        raise SystemExit("No Any2Full .pt2 under weights/any2full")
    model = Any2Full_PT2(ckpt, device="cpu")
    path = str(_frames(frames_dir, 1)[0])
    # Synthetic prompt depth: zeros + a few anchors (preprocess only needs it
    # finite; the model is not run here).
    rgb = np.asarray(Image.open(path).convert("RGB"))
    h, w = rgb.shape[:2]
    depth = np.zeros((h, w), dtype=np.float32)
    depth[h // 4 : h // 2, w // 4 : w // 2] = 0.5
    rgb_t, dep_t = model.preprocess(path, depth)
    return [("any2full_rgb:fp32", _np(rgb_t)), ("any2full_dep:fp32", _np(dep_t))]


def _probe_tapip3d(frames_dir: Path):
    from utils.streaming_utils import load_batch_frames

    paths = _frames(frames_dir)
    file_list = list(enumerate(paths))
    out = load_batch_frames(file_list, 0, len(paths))
    return [("tapip3d_frames:lsb", _align(_np(out)))]


_PROBES = {
    "da3": _probe_da3,
    "vggt": _probe_vggt,
    "romav2": _probe_romav2,
    "sam3": _probe_sam3,
    "any2full": _probe_any2full,
    "tapip3d": _probe_tapip3d,
}


# ---------------------------------------------------------------------------
# e2e probes: real model runs (da3 nested + vggt) when weights are present.
# mode=report entries are saved/printed but never asserted.
# ---------------------------------------------------------------------------

def _e2e_weights() -> dict:
    """Checkpoints for the e2e probes, resolved identically at capture and
    compare time so both runs target the same models."""
    return {
        "da3_av": _first_pt2(str(DEFAULT_WEIGHTS / "da3"), "*anyview*.pt2"),
        "da3_mt": _first_pt2(str(DEFAULT_WEIGHTS / "da3"), "*metric*.pt2"),
        "vggt": _first_pt2(str(DEFAULT_WEIGHTS / "vggt_omg")),
    }


def _probe_e2e(frames_dir: Path, weights: dict) -> list:
    items = []
    if weights["da3_av"] and weights["da3_mt"]:
        from depth_models.da3.model.da3nested import DA3NestedPT2

        model = DA3NestedPT2(weights["da3_av"], weights["da3_mt"], compile=False)
        imgs = [str(p) for p in _frames(frames_dir, model.av.num_views)]
        out = model.infer(imgs)
        items += [(f"e2e_da3_{k}:report", np.asarray(v)) for k, v in out.items()]
    if weights["vggt"]:
        from depth_models.vggt_omega.vggt_omega import VGGT_Omega

        model = VGGT_Omega(weights["vggt"])
        imgs = [str(p) for p in _frames(frames_dir, model.num_views)]
        out = model.infer(imgs)
        items += [(f"e2e_vggt_{k}:report", np.asarray(v)) for k, v in out.items()]
    return items


# ---------------------------------------------------------------------------
# capture / compare
# ---------------------------------------------------------------------------

def _save(golden_dir: Path, runtime: str, items) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    np.savez(golden_dir / f"{runtime}.npz", **{k: v for k, v in items})


def _load(golden_dir: Path, runtime: str) -> dict:
    with np.load(golden_dir / f"{runtime}.npz") as z:
        return {k: z[k] for k in z.files}


def _check(key: str, mode: str, golden: np.ndarray, new: np.ndarray) -> tuple:
    golden, new = _align(golden), _align(new)
    if golden.shape != new.shape:
        return False, f"shape {golden.shape} != {new.shape}"
    if golden.dtype.kind in "ui" and new.dtype.kind == "f":
        new = new.astype(golden.dtype)
    diff = np.abs(golden.astype(np.float64) - new.astype(np.float64))
    mx = float(diff.max())
    p99 = float(np.percentile(diff, 99.9))
    if mode == "exact":
        return mx == 0.0, f"max={mx:.3g}"
    if mode == "lsb":
        return p99 <= 1.0 and mx <= 3.0, f"p99={p99:.3g} max={mx:.3g}"
    if mode == "fp32":
        return mx <= 2e-2, f"max={mx:.3g}"
    if mode == "report":
        return True, f"REPORT max={mx:.3g} p99={p99:.3g}"
    return False, f"bad mode {mode}"


def run(args) -> int:
    frames_dir = Path(args.frames_dir)
    runtimes = list(_PROBES) if args.runtime == "all" else [args.runtime]
    ok = True
    for rt in runtimes:
        probe = _PROBES[rt]
        if args.mode == "capture":
            items = probe(frames_dir)
            _save(Path(args.golden_dir), rt, items)
            print(f"[capture] {rt}: {len(items)} entries -> {args.golden_dir}/{rt}.npz")
            continue
        items = probe(frames_dir)
        goldens = _load(Path(args.golden_dir), rt)
        rt_ok = True
        for key, val in items:
            name, _, mode = key.rpartition(":")
            if name not in goldens:
                print(f"[compare] {rt}: MISSING golden for {name}")
                rt_ok = False
                continue
            passed, msg = _check(key, mode, goldens[name], val)
            print(f"[compare] {rt}: {'PASS' if passed else 'FAIL'} {name} ({mode}) {msg}")
            rt_ok = rt_ok and passed
        print(f"[compare] {rt}: {'PASS' if rt_ok else 'FAILED'}")
        ok = ok and rt_ok
    # e2e: real-model runs (da3 nested / vggt) whenever their weights exist.
    weights = _e2e_weights()
    if weights["vggt"] or (weights["da3_av"] and weights["da3_mt"]):
        items = _probe_e2e(frames_dir, weights)
        if args.mode == "capture":
            _save(Path(args.golden_dir), "e2e", items)
            print(f"[capture] e2e: {len(items)} entries (real model runs)")
        else:
            goldens = _load(Path(args.golden_dir), "e2e")
            for key, val in items:
                name, _, mode = key.rpartition(":")
                if name not in goldens:
                    print(f"[compare] e2e: MISSING golden for {name}")
                    ok = False
                    continue
                passed, msg = _check(key, mode, goldens[name], val)
                print(f"[compare] e2e: {'PASS' if passed else 'FAIL'} {name} {msg}")
    elif args.mode == "compare":
        print("[compare] e2e: skipped (no da3/vggt weights under weights/)")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Golden capture/compare for image preprocess")
    p.add_argument("--mode", choices=["capture", "compare"], required=True)
    p.add_argument("--runtime", choices=list(_PROBES) + ["all"], default="all")
    p.add_argument("--golden-dir", default=str(GOLDEN_DEFAULT))
    p.add_argument("--frames-dir", default=str(ASSETS / "head_rgbd"))
    args = p.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
```

`scripts/general_test/verify_image_inputs.sh`:

```bash
#!/usr/bin/env bash
# Golden capture/compare for the image-input refactor (spec 2026-09-02).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-capture}"   # capture | compare
RUNTIME="${2:-all}"    # da3|vggt|romav2|sam3|any2full|tapip3d|all
uv run python tools/general_test/module/compare_image_inputs.py \
    --mode "$MODE" --runtime "$RUNTIME"
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/general_test/verify_image_inputs.sh`
Expected: no output.

- [ ] **Step 3: Capture the baseline on current (pre-refactor) code**

Run: `bash scripts/general_test/verify_image_inputs.sh capture`
Expected: six `[capture] <runtime>: N entries` lines plus `[capture] e2e: …` (real da3/vggt runs — takes a minute or two); npz files exist under `output/verify_image_inputs/`.

- [ ] **Step 4: Self-check the harness (compare against the fresh goldens)**

Run: `bash scripts/general_test/verify_image_inputs.sh compare`
Expected: every entry `PASS` (identical code → zero diffs), e2e `REPORT max=0`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add tools/general_test/module/compare_image_inputs.py \
        scripts/general_test/verify_image_inputs.sh
git commit -m "test: golden capture/compare harness for image preprocess

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Shared helpers in `image_io.py` + unit tests (TDD)

**Files:**
- Modify: `src/utils/image_io.py`
- Create: `tests/test_image_io.py`

**Interfaces:**
- Consumes: existing `ImageInput`, `to_image_tensor`.
- Produces: `to_pixel_uint8(x: torch.Tensor) -> torch.Tensor`, `letterbox(x: torch.Tensor, target_h: int, target_w: int, *, scale_mode: Literal["trunc2","round"]="trunc2", resize=v2.InterpolationMode.BILINEAR, antialias: bool=False) -> tuple[torch.Tensor, dict]`, `imagenet_normalize(x: torch.Tensor) -> torch.Tensor`. Every runtime's `_load_image` is `to_pixel_uint8(to_image_tensor(image))`.

- [ ] **Step 1: Write the failing tests**

`tests/test_image_io.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --group dev pytest tests/test_image_io.py -q`
Expected: collection errors (`letterbox`/`imagenet_normalize`/`to_pixel_uint8` do not exist).

- [ ] **Step 3: Implement the helpers**

`src/utils/image_io.py` — extend the module docstring's first line list, add imports (`math`, `torch.nn.functional`), and append after `to_image_tensor`:

```python
def to_pixel_uint8(x: torch.Tensor) -> torch.Tensor:
    """Rescale a CHW float [0,1] tensor into uint8 pixel space [0,255].

    ``to_image_tensor`` passes float [0,1] CHW tensors through untouched; this
    is the uniform "decoded image" representation the runtimes consume. uint8
    input passes through unchanged (same object).
    """
    if x.dtype == torch.uint8:
        return x
    return (x.clamp(0.0, 1.0) * 255.0).round().byte()


def letterbox(
    x: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    scale_mode: str = "trunc2",
    resize: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
    antialias: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Aspect-preserving center-pad of a CHW image to ``(target_h, target_w)``.

    ``x`` is a CHW 3-channel pixel-space image (uint8, or float [0,1] —
    rescaled to uint8 on entry via :func:`to_pixel_uint8`).  Resize runs on
    uint8 (integer quantization preserved, like the legacy cv2/PIL letterboxes);
    the zero pad is centered.  Returns ``(padded uint8 CHW, meta)`` with meta
    keys ``orig_h/orig_w/scale_factor/tile_h/tile_w/pad_top/pad_left`` computed
    in float64 — bit-identical to the legacy math.

    ``scale_mode``: ``"trunc2"`` truncates the uniform scale to 2 decimals
    (DA3 / WAFT convention); ``"round"`` keeps the raw scale and rounds the
    tile size (VGGT convention).  ``resize``/``antialias`` select the
    torchvision resampling.
    """
    x = to_pixel_uint8(x)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(
            f"letterbox expects a CHW 3-channel image, got shape {tuple(x.shape)}"
        )
    orig_h, orig_w = int(x.shape[1]), int(x.shape[2])
    raw_scale = min(target_w / orig_w, target_h / orig_h)
    if scale_mode == "trunc2":
        scale = math.floor(raw_scale * 100.0) / 100.0
        if scale <= 0:
            scale = raw_scale
        tile_w, tile_h = int(orig_w * scale), int(orig_h * scale)
    elif scale_mode == "round":
        scale = raw_scale
        tile_w, tile_h = round(orig_w * scale), round(orig_h * scale)
    else:
        raise ValueError(f"unknown scale_mode {scale_mode!r} (trunc2 | round)")

    pad_w, pad_h = target_w - tile_w, target_h - tile_h
    pad_left, pad_top = pad_w // 2, pad_h // 2
    resized = v2.functional.resize(
        x, (tile_h, tile_w), interpolation=resize, antialias=antialias
    )
    padded = F.pad(resized, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top))
    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "scale_factor": float(scale),
        "tile_h": tile_h,
        "tile_w": tile_w,
        "pad_top": int(pad_top),
        "pad_left": int(pad_left),
    }
    return padded, meta


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """ImageNet-normalize a uint8 pixel-space CHW tensor: ``(x/255-mu)/std`` fp32."""
    f = x.float().div(255.0)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (f - mean.to(f.device)) / std.to(f.device)
```

Also update the module docstring (`ImageInput` list stays; add one line: "``letterbox`` / ``imagenet_normalize`` / ``to_pixel_uint8`` provide the shared pixel-space geometry for the model runtimes").

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --group dev pytest tests/test_image_io.py -q`
Expected: all pass (the cv2 cross-checks confirm the ≤1 LSB / ≤3 max drift bounds).

- [ ] **Step 5: Commit**

```bash
git add src/utils/image_io.py tests/test_image_io.py
git commit -m "feat: shared image_io letterbox/imagenet_normalize/to_pixel_uint8 helpers

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: DA3 tensor-first rewrite (base_da3 → anyview → metric → nested)

**Files:**
- Modify: `src/depth_models/da3/model/base_da3.py`, `da3anyview.py`, `da3metric.py`, `da3nested.py`

**Interfaces:**
- Consumes: `to_image_tensor`, `to_pixel_uint8`, `letterbox`, `imagenet_normalize` from Task 2.
- Produces: `BaseDA3Model._load_image(image: ImageInput) -> torch.Tensor` (CHW uint8 CPU); `preprocess_views(imgs) -> tuple[torch.Tensor, list[dict]]` (CPU fp32 `(1,N,3,H,W)`); `DA3AnyViewPT2.run(img_batch: torch.Tensor) -> dict[str, np.ndarray]`; `DA3MetricPT2.run(feed: dict[str, torch.Tensor])`, `infer_view(img: torch.Tensor)`. Outputs stay numpy; meta exact.

- [ ] **Step 1: Rewrite the preprocessing half of `base_da3.py`**

Replace the module docstring's "``cv2`` is imported lazily…" sentence, delete `import numpy as np` only if unused after edits (it still is — `crop_to_tile`/`map_anyview_keys`/`extract_metric`/`apply_mono_sky`/`align_with_metric` use numpy), delete `_MEAN`/`_STD` class attrs, and add the image_io import at the top:

```python
import numpy as np
import torch
from depth_models.da3.utils.alignment import align_anyview_with_metric
from utils.image_io import (
    ImageInput,
    imagenet_normalize,
    letterbox,
    to_image_tensor,
    to_pixel_uint8,
)
```

Replace `_preprocess_one` and add `_load_image` (keep `preprocess_views` structure, but the batch is a tensor):

```python
    def _load_image(self, image: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(image))

    def preprocess_views(
        self,
        imgs: list,
        target_h: int | None = None,
        target_w: int | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Resize/normalize *N* views to ``(1, N, 3, H, W)``.

        ``target_h``/``target_w`` default to ``self.target_h``/``self.target_w``;
        pass explicit values to preprocess for a differently-sized model (e.g.
        the metric branch inside the nested pipeline).  Returns the padded
        fp32 CPU tensor batch and per-view crop-back metadata.
        """
        th = target_h if target_h is not None else self.target_h
        tw = target_w if target_w is not None else self.target_w
        n = len(imgs)
        proc = torch.zeros((n, 3, th, tw), dtype=torch.float32)
        metas: list[dict] = []
        for i in range(n):
            proc[i], meta = self._preprocess_one(imgs[i], th, tw)
            metas.append(meta)
        return proc[None], metas  # add B=1

    def _preprocess_one(
        self,
        img: ImageInput,
        target_h: int,
        target_w: int,
    ) -> tuple[torch.Tensor, dict]:
        """Letterbox a single view (any ImageInput; numpy arrays are RGB):
        aspect-preserving resize with a 2-decimal-truncated scale, then
        center-pad to the target.  Returns CHW fp32 (ImageNet-normalized) and
        per-view meta for the crop-back on post-process.
        """
        x = self._load_image(img)
        padded, meta = letterbox(x, target_h, target_w, scale_mode="trunc2")
        return imagenet_normalize(padded), meta
```

Then confirm no stale references to the removed constants remain:
Run: `grep -rn "_MEAN\|_STD" src/depth_models | grep -v image_io`
Expected: no output.

- [ ] **Step 2: `da3anyview.py` — run() takes the tensor batch; docs RGB**

Change the class docstring line "images → depth bundle (numpy in/out)" to "(ImageInput in, numpy out)". Replace `run`:

```python
    def run(self, img_batch: torch.Tensor) -> dict[str, np.ndarray]:
        """Run the exported graph on the preprocessed ``(1,N,3,H,W)`` fp32
        CPU tensor batch."""
        image = img_batch.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return dict(zip(self._OUTPUT_NAMES, [o.float().cpu().numpy() for o in outputs]))
```

`infer`'s type hint becomes `imgs: list` unchanged, docstring "image paths or BGR arrays" → "image paths, RGB arrays or tensors". Add `torch` import if not already present (it is).

- [ ] **Step 3: `da3metric.py` — tensor feed**

Replace `run` and `infer_view`:

```python
    def run(self, feed: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Run the exported graph: ``{"image": (1,3,H,W) fp32 tensor}``."""
        image = feed["image"].to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return dict(zip(self._OUTPUT_NAMES, [o.float().cpu().numpy() for o in outputs]))

    def infer_view(
        self, img: torch.Tensor, apply_mono_sky: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``img`` is a preprocessed ``(3, H, W)`` CHW fp32 tensor →
        ``(depth, sky)`` numpy."""
        raw = self.run({"image": img.unsqueeze(0)})
        depth, sky = self.extract_metric(raw)
        if apply_mono_sky:
            depth = self.apply_mono_sky(depth, sky)
        return depth, sky
```

Update the class docstring of `_run_metric_branch` callers accordingly (docstring "single already-preprocessed view" stays valid). Drop the now-unused `feed["image"].astype` path.

- [ ] **Step 4: `da3nested.py` — tensor slicing in the metric branch**

In `infer`, the line `img_batch, metas = self.av.preprocess_views(imgs)` now yields a tensor (no change needed). In `_run_metric_branch`, change the signature hint `av_img_batch: np.ndarray` → `torch.Tensor` and `n = av_img_batch.shape[1]` stays; the per-view slice `av_img_batch[0, i]` is now a tensor view — pass it straight to `metric.infer_view(...)`. Add `import torch`. Nothing else changes.

- [ ] **Step 5: Golden gate — compare DA3**

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime da3`
Expected: `da3_batch:fp32` max ≤ 2e-2 (typically 0 — same fp32 op order), all `da3_meta*:exact` max = 0, exit 0.

- [ ] **Step 6: Unit tests still green**

Run: `uv run --group dev pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/depth_models/da3/model/base_da3.py \
        src/depth_models/da3/model/da3anyview.py \
        src/depth_models/da3/model/da3metric.py \
        src/depth_models/da3/model/da3nested.py
git commit -m "refactor: DA3 tensor-first preprocess via shared image_io helpers

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: VGGT-Omega rewrite (base + wrapper)

**Files:**
- Modify: `src/depth_models/vggt_omega/base_vggt_omega.py`, `src/depth_models/vggt_omega/vggt_omega.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces: `BaseVGGTOmega._load_image(image) -> torch.Tensor` (CHW uint8); `infer` unchanged signature (`list[ImageInput]`); `build_feed(canvases: list[torch.Tensor]) -> dict[str, torch.Tensor]` (CPU fp32 `(1,N,3,H,W)`); `VGGT_Omega._forward(feed)` consumes the tensor feed. numpy outputs unchanged.

- [ ] **Step 1: `base_vggt_omega.py` — delete module loaders, add `_load_image`, tensor canvases**

Delete the module functions `_load_rgb_uint8` (lines ~130-139) and `_letterbox_preprocess` (lines ~83-103) and the `from PIL import Image` import (now unused). Keep `quat_to_mat`, `encoding_to_camera`, `_adjust_intrinsics`, `_crop_spatial`, `postprocess` untouched.

Imports: `numpy as np`, `torch`, and `from torchvision.transforms import v2`; add

```python
from utils.image_io import (
    ImageInput,
    letterbox,
    to_image_tensor,
    to_pixel_uint8,
)
```

Add the loader method and rewrite `infer` + `build_feed`:

```python
    def _load_image(self, image: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(image))

    def infer(self, images: list[ImageInput]) -> dict[str, np.ndarray]:
        """Run inference on image paths, RGB arrays or tensors.

        The static image size comes from the backend base's resolved geometry;
        images are letterboxed (VGGT convention: raw-scale ``round`` tiles,
        bicubic) to it and spatial outputs are cropped back to the content
        region.  Mean/std normalization happens inside the model, so inputs
        only need to be float32 RGB in [0,1] (done here).

        Returns ``{pose_enc, extrinsic, intrinsic, depth, depth_conf, crop}``.
        """
        if not images:
            raise ValueError("At least one image is required")
        canvases, metas = zip(
            *(
                letterbox(
                    self._load_image(image),
                    self.target_h,
                    self.target_w,
                    scale_mode="round",
                    resize=v2.InterpolationMode.BICUBIC,
                )
                for image in images
            )
        )
        # VGGT pads into float [0,1] canvases; the letterbox meta maps 1:1 onto
        # the old (ox, oy, h, w) crop tuple (pad_left, pad_top, tile_h, tile_w).
        canvases = [c.float().div(255.0) for c in canvases]
        crops_np = np.asarray(
            [
                [m["pad_left"], m["pad_top"], m["tile_h"], m["tile_w"]]
                for m in metas
            ],
            dtype=np.int64,
        )
        outputs = self._forward(self.build_feed(canvases))
        return self.postprocess(outputs, crops_np)

    def build_feed(self, canvases: list[torch.Tensor]) -> dict:
        """Stack the letterboxed CHW canvases into the model's ``(1, N, 3, H, W)``
        fp32 CPU tensor, keyed by the model's own input name.  Each backend
        casts to the model's expected dtype (fp16 export / declared ONNX
        dtype)."""
        batch = torch.stack(canvases)[None]
        return {self.inputs[0]["name"]: batch}
```

- [ ] **Step 2: `vggt_omega.py` — `_forward` consumes the tensor feed**

Replace `_forward` and update the class docstring input description:

```python
    def _forward(self, feed: dict) -> dict[str, np.ndarray]:
        """Run the exported graph on the ``(1, N, 3, H, W)`` CPU tensor feed,
        casting to the export's input dtype; returns the raw ``(1, N, ...)``
        outputs keyed for ``BaseVGGTOmega.postprocess``."""
        batch = next(iter(feed.values()))
        dtype = torch.float16 if self._input_fp16 else torch.float32
        with torch.inference_mode():
            pose_e, depth_e, conf_e = self._graph_module(
                batch.to(dtype=dtype, device="cuda")
            )
        return {
            "pose_enc": pose_e.float().cpu().numpy(),
            "depth": depth_e.float().cpu().numpy(),
            "depth_conf": conf_e.float().cpu().numpy(),
        }
```

(`infer` on the wrapper still enforces `len(images) == self.num_views`; its type hint becomes `list[ImageInput]`.)

- [ ] **Step 3: Golden gate — compare VGGT**

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime vggt`
Expected: `vggt_feed_*:fp32` max ≤ 2e-2 (bicubic kernel change only), crops exact, exit 0.

- [ ] **Step 4: Unit tests still green** — `uv run --group dev pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/depth_models/vggt_omega/base_vggt_omega.py \
        src/depth_models/vggt_omega/vggt_omega.py
git commit -m "refactor: VGGT-Omega tensor-first letterbox via shared image_io

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: RoMaV2 — decode-only `_load_image`

**Files:**
- Modify: `src/det_seg_models/romav2/romav2.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces: `RoMaV2PT2._load_image(img_like: ImageInput) -> torch.Tensor` (CHW uint8 CPU, keeps the `I;16` guard); `match_pair`/`match` signatures unchanged; float/255 + batching + device move into `match_pair`.

- [ ] **Step 1: Rewrite the loader and `match_pair`**

Replace the local `ImageInput` alias (line ~45) with an import, and update the module docstring's dependency note:

```python
from utils.image_io import ImageInput, to_image_tensor, to_pixel_uint8
```

Replace `_load_image`:

```python
    def _load_image(self, img_like: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU).

        numpy arrays must be uint8 RGB (HWC); CHW tensors may be uint8 or
        float [0,1] (rescaled to uint8 pixel space).
        """
        if isinstance(img_like, (str, Path)):
            with Image.open(img_like) as im:
                mode = im.mode
        elif isinstance(img_like, Image.Image):
            mode = img_like.mode
        else:
            mode = None
        if mode == "I;16":
            raise NotImplementedError("Can't handle 16 bit images")
        return to_pixel_uint8(to_image_tensor(img_like))
```

In `match_pair`, replace the two `_load_image` calls and keep the rest:

```python
        img_A = self._load_image(img_A).float().div(255.0).unsqueeze(0).to(self.device)
        img_B = self._load_image(img_B).float().div(255.0).unsqueeze(0).to(self.device)
```

Update `match`'s docstring: "(uint8 or float)" numpy wording → uint8 RGB arrays or tensors; tensors uint8 or float [0,1]. `np.ndarray` import stays (masks etc.). PIL import stays.

- [ ] **Step 2: Golden gate — compare RoMaV2** (needs CUDA)

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime romav2`
Expected: all `romav2_*:fp32` pass (max ≤ 2e-2; loads are decode-identical so diffs ≈ 0), exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/det_seg_models/romav2/romav2.py
git commit -m "refactor: RoMaV2 decode-only _load_image via shared image_io

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: SAM3 — uniform `_load_image` (behavior-neutral)

**Files:**
- Modify: `src/det_seg_models/sam3/sam3_model.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces: `Sam3Image._load_image(image: ImageInput) -> torch.Tensor`; `preprocess_image` and `predict` route through it.

- [ ] **Step 1: Add `_load_image` and route both call sites**

Extend the existing `from utils.image_io import ...` line with `to_pixel_uint8`, and add:

```python
    def _load_image(self, image: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(image))
```

In `preprocess_image`, replace the `to_image_tensor(image)` call with `self._load_image(image)`, and update its docstring ("float [0,1] CHW tensors are rescaled to uint8 on entry"). In `predict`, replace `image_t = to_image_tensor(image)` with `image_t = self._load_image(image)` (the h/w and the reused-tensor flow stay identical — `preprocess_image` re-running `_load_image` on the uint8 tensor is a passthrough).

- [ ] **Step 2: Golden gate — compare SAM3**

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime sam3`
Expected: `sam3_preprocess:fp32` max = 0 (path input, unchanged transform), exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/det_seg_models/sam3/sam3_model.py
git commit -m "refactor: SAM3 image entry via uniform _load_image

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Any2Full — `_load_image` instance method, ImageInput param

**Files:**
- Modify: `src/depth_models/a3f/any2full.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces: `Any2Full_PT2._load_image(rgb: ImageInput) -> torch.Tensor`; `preprocess(rgb: ImageInput, depth_metrics: np.ndarray, denoise=False, denoise_kwargs=None)` — same values/op order as today.

- [ ] **Step 1: Rewrite the loader + preprocess**

Imports: add `to_pixel_uint8` and `imagenet_normalize` to the existing `from utils.image_io import ...`; drop `torchvision.transforms as T` if unused elsewhere (it is only used for `Normalize` here) — keep the import if `T.` appears elsewhere (it does not in this file).

Delete the module-level `_load_rgb` function and replace `preprocess` (add `_load_image` above it):

```python
    def _load_image(self, rgb: ImageInput) -> torch.Tensor:
        """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
        return to_pixel_uint8(to_image_tensor(rgb))

    def preprocess(self, rgb: ImageInput, depth_metrics: np.ndarray,
                   denoise: bool = False, denoise_kwargs=None) -> tuple:
        rgb = imagenet_normalize(self._load_image(rgb)).unsqueeze(0).to(
            device=self.device, dtype=self.dtype
        )
        if denoise:
            depth_metrics = remove_outliers(depth_metrics, **(denoise_kwargs or {}))
        # Input sanity check (mirrors the former in-graph assert in
        # Any2Full.forward, which was removed so forward stays exportable).
        if not np.isfinite(depth_metrics).all():
            raise ValueError(f"Input depth contains nan/inf")
        rgb = self._check_and_resize(rgb, (self.target_h, self.target_w), "bilinear", "rgb")
        dep = torch.from_numpy(depth_metrics).unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        dep = self._check_and_resize(dep, (self.target_h, self.target_w), "nearest", "depth")
        return rgb, dep
```

(Op order preserved: old `_load_rgb` normalized on CPU then moved device/dtype, then resized. `imagenet_normalize` returns fp32 CPU; `.to(dtype=...)` handles the bf16 graph.)

- [ ] **Step 2: Golden gate — compare Any2Full**

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime any2full`
Expected: `any2full_rgb:fp32`/`any2full_dep:fp32` max ≈ 0 (identical op order), exit 0.

- [ ] **Step 3: Commit**

```bash
git add src/depth_models/a3f/any2full.py
git commit -m "refactor: Any2Full tensor-first preprocess via shared image_io

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: TAPIP3D frame loaders in `streaming_utils.py`

**Files:**
- Modify: `src/utils/streaming_utils.py`

**Interfaces:**
- Consumes: Task 2 helpers.
- Produces: `load_batch_frames(...) -> torch.Tensor` CHW uint8 `(T,3,H,W)` CPU (was `(T,H,W,3)` numpy); `load_resized_batch` same output contract `(T,3,H,W)` float [0,1] CPU tensor; `load_pair` same `(N,H,W,3)` uint8 numpy contract.

- [ ] **Step 1: Rewrite `load_batch_frames` and `load_resized_batch`**

Top of file: add `import torch` if absent, and `from utils.image_io import to_image_tensor`; add `from torchvision.transforms.v2 import InterpolationMode` (or import the module and reference `v2.InterpolationMode`).

Replace `load_batch_frames` (lines ~132-140):

```python
def load_batch_frames(file_list, start, end):
    """Load a slice of frames into a (T, 3, H, W) uint8 CHW CPU tensor."""
    return torch.stack(
        [to_image_tensor(fpath) for _, fpath in file_list[start:end]]
    )
```

In `load_resized_batch`, replace the whole body from `video_np = load_batch_frames(...)` through the `return` statement (lines ~280-302) with:

```python
    video = load_batch_frames(file_list, start, end)          # (T,3,H0,W0) uint8
    geo = load_npz_batch(npz_dir, file_list, start, end)
    orig_h, orig_w = video.shape[1:3]

    video_rs = torch.stack([
        F_resize(video[t], (inference_h, inference_w),
                 interpolation=InterpolationMode.BILINEAR, antialias=False)
        for t in range(video.shape[0])])
    depths_rs = np.stack([
        resize_depth_bilinear(geo["depth"][t], (inference_w, inference_h))
        for t in range(geo["depth"].shape[0])])

    scale_y = (inference_h - 1) / (orig_h - 1)
    scale_x = (inference_w - 1) / (orig_w - 1)
    intrs = geo["intrs"].copy()
    intrs[:, 0, :] *= scale_x
    intrs[:, 1, :] *= scale_y

    video_t = video_rs.float() / 255.0
    depths_t = torch.from_numpy(depths_rs).float()
    intrs_t = torch.from_numpy(intrs).float()
    extrs_t = torch.from_numpy(geo["extrs"]).float()
    return video_t, depths_t, intrs_t, extrs_t
```

with `from torchvision.transforms import v2 as _v2` and `F_resize = _v2.functional.resize` alias at the top (cv2 stays in this module for the depth/stream helpers — do not remove it).

- [ ] **Step 2: `load_pair` decodes through image_io**

In `load_pair`, replace the cv2 decode + resize block (lines ~114-121) with:

```python
        rgb = (
            _v2.functional.resize(
                to_image_tensor(img_path), (H, W),
                interpolation=InterpolationMode.BILINEAR, antialias=False,
            )
            .permute(1, 2, 0)
            .numpy()
        )
```

Output remains `(N,H,W,3)` uint8 RGB numpy. Update its docstring ("uint8 RGB").

- [ ] **Step 3: Golden gate — compare TAPIP3D**

Run: `uv run python tools/general_test/module/compare_image_inputs.py compare --runtime tapip3d`
Expected: `tapip3d_frames:lsb` p99 ≤ 1, max ≤ 3 (cv2→PIL decode + uint8 resize drift), exit 0.

- [ ] **Step 4: Unit tests still green** — `uv run --group dev pytest tests/ -q` → all pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/streaming_utils.py
git commit -m "refactor: TAPIP3D frame loaders decode via image_io (tensor-first)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: A2F-nested hints + online streamer stores RGB

**Files:**
- Modify: `src/depth_models/a3f/a2fnested.py`, `tools/astribot/run_step2_depth_stream.py`

**Interfaces:**
- Consumes: Task 3/7 runtime changes.
- Produces: no API change — behavior change: dataset frames are cached RGB; the untouched WAFT motion-mask path flips to BGR at its call site.

- [ ] **Step 1: `a2fnested.py` docstring + hints**

- `infer(self, imgs: list[str | np.ndarray], ...)` → `list[ImageInput]` (import `ImageInput` from `utils.image_io`); docstring "paths or BGR arrays" → "paths, RGB arrays or tensors".
- In `_run_metric_branch`, update the `np.ascontiguousarray` comment (line ~158-161): it now guards array views for `torch.from_numpy` inside the depth map handling only — the RGB view is handled by image_io. Keep the call as-is (still correct and harmless).

- [ ] **Step 2: `run_step2_depth_stream.py` — RGB cache, drop `_rgb_input`, WAFT flips locally**

1. `OnlineStreaming` class comment (lines ~177-179): replace

```python
    #: frames are stored BGR (dataset PIL RGB flipped once at fetch); models
    #: expecting RGB (VGGT-Omega) flip back when assembling a chunk
    _rgb_input = False
```

with

```python
    #: frames are stored RGB (dataset PIL RGB kept as-is at fetch); every
    #: model consumes RGB.  Only the WAFT motion-mask path (untouched, still
    #: BGR-based) flips back at its call site (_compute_step_masks).
```

2. In `_process_chunk` (lines ~243-248), delete the flip:

```python
        frames = []
        for i, t in enumerate(steps):
            arr = self._frame_cache[t][self._cam_keys[i % self.num_cams]]
            if self._rgb_input:
                arr = arr[:, :, ::-1]
            frames.append(arr)
```

becomes

```python
        frames = []
        for i, t in enumerate(steps):
            frames.append(self._frame_cache[t][self._cam_keys[i % self.num_cams]])
```

3. Subclass blocks (lines ~317-335): remove each `_rgb_input = ...` line; update docstrings:

```python
class OnlineVGGTStreaming(OnlineStreaming, VGGT_OMG_Streaming):
    """VGGT-Omega streaming with online frames; RGB arrays (all runtimes
    consume RGB)."""


class OnlineDA3Streaming(OnlineStreaming, DA3_Streaming):
    """DA3 streaming with online frames; RGB arrays."""


class OnlineA2FStreaming(OnlineStreaming, A2F_Streaming):
    """Any2Full RGB-D streaming with online frames; RGB arrays and the raw
    depth comes from the dataset (the a2f backend's depth_dirs are virtual —
    see SubtaskStreamExtract._ensure_stream)."""
```

4. `_compute_step_masks` (lines ~301-310): feed WAFT BGR (contiguous copies):

```python
    def _compute_step_masks(self, t: int) -> list[np.ndarray]:
        """One motion mask per camera for step t: WAFT flow between the RGB
        frames (t, t + stride), flipped to BGR for the (untouched, BGR-based)
        WAFT model.  255 = moving, 0 = static (the stacker above inverts to
        1 = static, matching the disk-mask convention)."""
        stride = self.wrapper.args.stride
        thr = self.wrapper.args.motion_threshold
        masks = []
        for cam_key in self._cam_keys:
            def _bgr(key, step):
                return np.ascontiguousarray(
                    self._frame_cache[step][key][:, :, ::-1]
                )
            flow = self.wrapper.waft_model(
                _bgr(cam_key, t), _bgr(cam_key, t + stride)
            )
            masks.append(_compute_motion_mask_gray(flow, thr))
        return masks
```

5. `_dataset_step` docstring/body (lines ~460-465): drop the flip so the cache holds RGB; update the comment:

```python
        frame = self._ensure_dataset()[t]
        step = {key: np.asarray(to_pil(frame[key]), dtype=np.uint8)
                for key in self.cam_keys.values()}
```

and its docstring: "RGB uint8 arrays (the dataset stores PIL RGB; no flip —
DA3/VGGT/a2f consume RGB; WAFT masks flip to BGR at the flow call) plus the
raw uint16 mm depth …".

- [ ] **Step 3: Static verification**

Run:

```bash
uv run python -m py_compile src/depth_models/a3f/a2fnested.py \
    tools/astribot/run_step2_depth_stream.py
grep -n "_rgb_input" tools/astribot/run_step2_depth_stream.py || echo "no _rgb_input left"
```

Expected: compile OK; grep prints nothing (or only the removed-flip comment) — the `_rgb_input` attribute and flip branch are gone.

- [ ] **Step 4: Commit**

```bash
git add src/depth_models/a3f/a2fnested.py \
        tools/astribot/run_step2_depth_stream.py
git commit -m "refactor: online streamer stores RGB; WAFT masks flip locally

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Full compare + end-to-end sanity + docs

**Files:**
- Modify: `CLAUDE.md`, `docs/general_test/module/waft.md` (only if it mentions `bgr_input`/`--no-bgr-input` — check first; otherwise leave), `README.md` (same check)

- [ ] **Step 1: Full golden compare**

Run: `bash scripts/general_test/verify_image_inputs.sh compare`
Expected: all six runtimes PASS, exit 0.

- [ ] **Step 2: End-to-end gate — real model runs diffed against the Task-1 e2e goldens**

The e2e capture ran real DA3-nested and VGGT forwards on the pre-refactor code (Task 1); compare re-runs them post-refactor. This catches wiring bugs in the changed `run()`/`_forward` signatures (missing `.to(device)`, dtype casts), which preprocess-level goldens cannot see.

Run: `bash scripts/general_test/verify_image_inputs.sh compare`
Expected: e2e entries `e2e_da3_depth`, `e2e_da3_depth_conf`, `e2e_da3_extrinsics`, `e2e_da3_intrinsics`, `e2e_vggt_*` all report near-zero `max`/`p99` (the capture ran the same checkpoints on the same GPU — expect exact or ~1e-6-level deltas). A large diff means the model feed or a signature wiring regressed — investigate before committing.

- [ ] **Step 3: Update CLAUDE.md**

- Package tree: `image_io.py  # ImageInput, to_image_tensor, letterbox, imagenet_normalize`
- "Model input/output conventions" section: rewrite the first bullets to state the RGB-uint8 contract, ImageInput acceptance, the tensor-first feeds, and the unified `_load_image`; add a line that WAFT is the legacy exception (BGR, numpy) until its `.pt2` replacement, and note the tests command `uv run --group dev pytest tests/` in Common commands.
- Any remaining "BGR array" wording in the DA3/VGGT/Any2Full hierarchy bullets → "ImageInput (RGB)".

- [ ] **Step 4: Grep docs for stale flags**

Run: `grep -rn "no_bgr_input\|bgr_input" README.md docs/ src/ --include="*.md" --include="*.py" | grep -v "flow_models/waft" | grep -v waft.md`
Expected: only `flow_models/waft/` hits remain (WAFT untouched). Fix any other hit (docs only).

- [ ] **Step 5: Final check + commit**

Run: `uv run --group dev pytest tests/ -q` → all pass. Then:

```bash
git add CLAUDE.md docs/general_test/module/waft.md README.md  # as applicable
git commit -m "docs: image input conventions (RGB uint8, tensor-first, _load_image)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 6: Report**

Summarize per-runtime compare maxima from the Step-1 output (expected ≤ 2e-2 fp32 / ≤ 3 LSB uint8, metas exact), note WAFT untouched, and list the follow-ups from spec §10.
