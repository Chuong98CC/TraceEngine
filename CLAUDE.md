# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Depth-estimation inference for two model families, each runnable on **two backends**
(ONNX Runtime and TensorRT):

- **Depth Anything 3 (DA3)** — monocular / multi-view. Three modules: `metric`
  (single-view depth + sky), `anyview` (multi-view depth, confidence, predicted
  cameras), and `nested` (any-view + metric + alignment — the full pipeline).
- **S2M2** — stereo metric depth (TensorRT only), with camera calibration for
  metric-scale output.

Backend-agnostic engine/session wrappers and camera types live in `src/base`;
model-specific pre/post-processing lives under `src/depth_model`. The repo uses
**uv** for environment management and **hatchling** as the build backend. Core
dependencies: TensorRT, ONNX Runtime, PyTorch, OpenCV, NumPy, trimesh. Requires
**Python ≥3.11, <3.13** (TensorRT / ONNX Runtime wheel availability). There are
no tests.

## Environment setup

```bash
# uv reads .python-version; installs the project editable (src/ on sys.path)
uv sync
```

The editable install puts both the repo root and `src/` on `sys.path`, so
`base`, `depth_model`, `utils`, and `tools` are all importable as top-level
packages.

## Package structure

```
src/
├── base/                       # Backend-agnostic wrappers + camera types
│   ├── base_trt.py             # TRTModel (ABC) + MonoDepthTRT / StereoDepthTRT
│   ├── base_onnx.py            # ONNXModel (ONNX Runtime session wrapper)
│   └── cam_structure.py        # CameraIntrinsics, CameraExtrinsics, read_calib_file
├── depth_model/
│   ├── da3/                    # Depth Anything 3
│   │   ├── base_da3.py         # BaseDA3Model mixin (preprocess / align / output-map)
│   │   ├── da3metric.py        # DA3MetricTRT / DA3MetricONNX
│   │   ├── da3anyview.py       # DA3AnyViewTRT / DA3AnyViewONNX
│   │   ├── da3nested.py        # DA3NestedTRT / DA3NestedONNX (compose anyview+metric)
│   │   └── utils/              # alignment.py, geometry.py, pose_align.py
│   └── s2m2/                   # S2M2 stereo
│       ├── s2m2.py             # S2M2Model(StereoDepthTRT)
│       └── export_onnx.py      # PyTorch → ONNX export (run inside the s2m2 repo)
└── utils/
    ├── astribot_dataloader.py  # Astribot dataset + calib loading, CAMERA_SETS
    ├── visualization.py        # save_depth_vis, export_glb (point cloud)
    ├── video_io.py             # open_video / write_video
    └── depth_utils.py          # pixel back-projection helpers
tools/
├── infer_da3_onnx_trt.py       # DA3 inference: metric|anyview|nested × onnx|trt
├── infer_s2m2.py               # S2M2 stereo inference (TRT)
└── export_trt.py               # ONNX → TensorRT engine via trtexec
scripts/
└── export_trt_docker.sh        # ONNX → TRT inside the nvcr.io/nvidia/tensorrt container
```

Model weights are expected under `weights/da3/` and `weights/s2m2/` (git-ignored).

## Class hierarchy

Backend bases (in `src/base`) are **image-count-agnostic** — they own engine/
session load, IO-tensor metadata, input-geometry resolution, and a name-matched
execution helper. They resolve both 4-D `(B,3,H,W)` mono/stereo/metric inputs and
5-D `(B,N,3,H,W)` any-view inputs.

- `TRTModel(ABC)` — TensorRT engine wrapper. Uses the TensorRT 10.3+
  `set_tensor_address` API (no manual CUDA memory management). `_run` binds
  inputs **by name**, allocates outputs, executes on the current CUDA stream.
  - `MonoDepthTRT` — single-image `_infer`
  - `StereoDepthTRT` — stereo pair `_infer` (dual-input or concatenated 6-channel)
- `ONNXModel` — ONNX Runtime session wrapper (CUDA EP + CPU fallback); `run(feed)`.

DA3 pre/post lives in one **mixin**, `BaseDA3Model` (needs `self.target_h/target_w`
from the backend base — no `__init__` of its own). It holds letterbox preprocessing
(ImageNet-normalized), extrinsics normalization, any-view feed assembly, output-key
mapping, mono-sky post-processing, and alignment orchestration.

Concrete DA3 wrappers **multiple-inherit** a backend base + the mixin, so each
model exists in both a TRT and an ONNX flavour:

- `DA3MetricTRT(TRTModel, BaseDA3Model)` / `DA3MetricONNX(ONNXModel, BaseDA3Model)`
- `DA3AnyViewTRT(TRTModel, BaseDA3Model)` / `DA3AnyViewONNX(ONNXModel, BaseDA3Model)`
- `DA3NestedTRT(BaseDA3Model)` / `DA3NestedONNX(BaseDA3Model)` — **compose** the
  any-view + metric wrappers (do not inherit a backend base themselves) and run
  the full nested pipeline.

Stereo:

- `S2M2Model(StereoDepthTRT)` — stereo metric depth via disparity → depth using
  `CameraIntrinsics`.

## Nested DA3 pipeline

`DA3Nested*.infer(imgs, extrs, intrs, ...)` reproduces `NestedDepthAnything3Net`:

1. **Any-view** — letterbox-preprocess N views, optionally feed camera pose
   priors, run, map outputs (`depth`/`depth_conf`/`extrinsics`/`intrinsics`).
2. **Metric branch** — per-view metric depth+sky on the *same* letterbox grid
   (metric engine must share the any-view input size; fail-fast otherwise).
3. **Align any-view → metric** (`align_anyview_with_metric`): scale any-view depth
   to metric, set sky regions to max depth.
4. **Optional Umeyama align to input poses** (`align_to_input_ext_scale`) — only
   when input extrinsics are provided. `align_scale=True` (default) replaces output
   poses with the input poses and rescales depth; `align_scale=False` keeps the
   predicted poses rigidly aligned into the input frame at the predicted scale.
5. Crop padded outputs back to each view's tile; un-pad the principal point.

When `extrs is None` the model predicts its own poses and step 4 is skipped —
matching `DepthAnything3.inference(extrinsics=None)`.

The `-with-camera-pose` any-view export additionally declares `extrinsics`/
`intrinsics` inputs. `BaseDA3Model.uses_extrinsics` reads the *loaded graph's*
inputs, so whether pose is fed is driven by the model, not a caller flag.

## Key data types (`src/base/cam_structure.py`)

- `CameraIntrinsics` (dataclass, slots) — `fx, fy, cx, cy, bl (baseline, metres),
  w, h`. Provides `from_intrinsics_matrix()`, `from_calib_file()`, per-axis scaled
  intrinsics, `disparity_to_depth()`, `back_project()` / `forward_project()`.
- `CameraExtrinsics` (dataclass, slots) — `R (3×3), T (3×1)`; `from_calib_file()`,
  `transform_pointcloud()`.
- `read_calib_file()` parses 3 JSON calibration layouts (see below), returning
  `Intrinsics`, `Extrinsics`, `Resolution`, `Baseline`, `Distortions` keyed by
  camera name.

## Common commands

```bash
# --- DA3 inference (ONNX or TensorRT) ---
# Nested pipeline, ONNX (default); model predicts its own poses
python tools/infer_da3_onnx_trt.py --module nested --camera-set set1 --frame 0

# Any-view branch, TensorRT
python tools/infer_da3_onnx_trt.py --backend trt --module anyview --frame 0

# Nested with camera-pose priors (selects the -with-camera-pose model + feeds poses)
python tools/infer_da3_onnx_trt.py --module nested --use-extrinsics --frame 0 --visualize

# Metric model, all frames
python tools/infer_da3_onnx_trt.py --module metric --all-frames

# --- S2M2 stereo inference (TensorRT) ---
python tools/infer_s2m2.py --frame_index 0 --save_depth --save_viz --save_glb

# --- Export ONNX → TRT engine ---
python tools/export_trt.py weights/da3/da3_metric_644x490_giant-large-1.1.onnx --precision fp16
# or inside the NVIDIA TensorRT container (avoids local env setup):
bash scripts/export_trt_docker.sh /abs/path/to/model.onnx fp16
```

`infer_da3_onnx_trt.py` auto-selects the camera set matching the loaded model's
view count (metric falls back to `set1`); an explicit `--camera-set` is honoured.
`--visualize` writes side-by-side depth JPEGs and, for anyview/nested, a point-cloud
`scene.glb`.

## Dataset loading

Inference tools read the **Astribot** dataset via `utils/astribot_dataloader.py`,
which hard-codes `DATA_ROOT = /home/chuong/workspace/demo_data` (the repo's `data`
symlink points here). `CAMERA_SETS` defines: `set0` (2 views, stereo L/R), `set1`
(3 views, head_rgbd + stereo L/R), `set2` (4 views, + torso_rgbd). Intrinsics are
scaled to 640×480 on load.

## TensorRT / backend version requirements

TensorRT code uses the 10.3+ `set_tensor_address` API. Engine files are
version-locked — on a deserialization failure, rebuild with the matching TensorRT
version via `scripts/export_trt_docker.sh`. Pinned runtimes:
`tensorrt-cu12==11.1.0.106`, `onnxruntime-gpu==1.28.0`.  TRT 11 removed the
`--fp16`/`--precisionConstraints`/`--layerPrecisions` trtexec flags (builds are
strongly typed by the ONNX's own dtypes); `tools/export_trt.py` builds with
`--decomposableAttentions='*'` — required for the fused opset-25 Attention nodes
in the DA3 ONNX exports (see [[trt11-export-attention-fix]]).  `--precision fp16`
pre-casts the ONNX via `tools/cast_onnx_fp16.py` (onnxconverter-common,
LayerNorm kept fp32); the docker script does the cast host-side since the
tensorrt container has no onnx tooling.  fp16 metric engine: ~6.9 ms vs
~36.9 ms tf32 on an RTX 5090, 644 MB vs 1310 MB.

## Model input/output conventions

- **DA3 preprocessing** (`BaseDA3Model`): BGR → RGB, aspect-preserving resize with
  a 2-decimal-truncated scale, center-pad (letterbox), ImageNet normalization
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Intrinsics are adjusted
  for the scale + pad. NCHW; any-view batches to `(1, N, 3, H, W)`.
- **DA3 metric**: outputs raw network `depth` + `sky`. Metric depth in metres is a
  **caller-side** step (`metric_depth = focal * depth / 300`, `focal = (fx+fy)/2`);
  the model graph carries no intrinsics. `apply_mono_sky` clamps sky-region depth to
  the non-sky 99th percentile (a NumPy replica of the step the ONNX graph omits) —
  applied for standalone metric inference, skipped inside the nested pipeline.
- **DA3 output-name resolution**: outputs are matched by keyword
  (`depth`/`depth_conf`/`pred_extrinsics`/`pred_intrinsics`/`sky`), so exact tensor
  names from the export don't matter.
- **StereoDepthTRT / S2M2**: two input modes — dual tensors (separate L/R) or a
  single concatenated 6-channel tensor. Right-view depth is obtained by mirroring
  the pair, running with the right image as reference, then mirroring back.
- All models use NCHW and GPU tensors; the wrappers handle HWC→CHW, batch dim, GPU
  transfer, and dtype matching the engine's expected input dtype.

## Calibration file formats

`read_calib_file()` handles three calibration JSON layouts:

1. **Full calibration**: `{"camera": {cam_name: {...}}, "lidar": ...}`
2. **Multi-camera**: `{cam_name: {...}}` (top-level keys are camera names)
3. **Single-camera**: flat dict with `resolution`/`intrinsics`/`extrinsics`/
   `distortions` at top level (camera name derived from the filename stem)

Stereo baselines are auto-derived from extrinsics when `parent_frame` names another
known camera (baseline in mm = L2 norm of translation, m→mm).
