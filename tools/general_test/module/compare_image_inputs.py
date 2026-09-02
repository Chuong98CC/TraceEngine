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
    if str(getattr(x, "dtype", "")) == "torch.bfloat16":
        x = x.float()  # numpy has no bf16 dtype (Any2Full preprocess is bf16)
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
    # npz keys are the bare entry names (mode tag stripped): compare resolves
    # goldens via key.rpartition(":")[0], so a ":fp32"-suffixed key would miss.
    np.savez(golden_dir / f"{runtime}.npz", **{k.rpartition(":")[0]: v for k, v in items})


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
