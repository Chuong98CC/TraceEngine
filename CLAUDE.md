# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Multi-model 3D-vision inference, each model deployed on the runtime that fits
it best (ONNX Runtime, TensorRT, TorchScript, or `torch.export` programs):

- **DA3 (Depth Anything 3)** — any-view depth + camera pose. Runs as two
  torch.export checkpoints (any-view + metric depth, fixed `num_views`,
  image-only input) composed by `DA3NestedPT2`, and as a streaming backend.
- **Any2Full (`a3f`)** — single-frame RGB-D densification: uses the RGB-D
  sensor's sparse metric depth as a **prompt** to ground a Depth Anything
  prediction, recovering metric scale with a deterministic affine fit
  (`torch.export` / `.pt2`).
- **VGGT-Omega** — multi-view depth + pose from `torch.export` programs; the
  main streaming backend.
- **Streaming** — chunked multi-camera depth+pose over long trajectories with
  SIM3 chunk alignment and optional loop closure.
- **WAFT** — dense optical flow (ONNX + TRT), used to build motion masks.
- **TAPIP3D** — 3D point tracking (ONNX), fixed query count.
- **SAM3** — promptable segmentation (`torch.export` / `.pt2`).

Backend-agnostic engine/session wrappers and camera types live in `src/base`;
model-specific pre/post-processing lives under `src/depth_models`,
`src/flow_models`, and `src/det_seg_models`. The repo uses **uv** for
environment management and **hatchling** as the build backend. Core
dependencies: TensorRT, ONNX Runtime, PyTorch, OpenCV, NumPy, trimesh. Requires
**Python ≥3.12, <3.13** (TensorRT / ONNX Runtime wheel availability). There are
no tests.

## Environment setup

```bash
# uv sync installs the project editable (src/ on sys.path)
uv sync
```

The editable install puts both the repo root and `src/` on `sys.path`, so
`base`, `depth_models`, `flow_models`, `det_seg_models`, `utils`, and `tools`
are all importable as top-level packages.

## Package structure

```
src/
├── base/                       # Backend-agnostic wrappers + camera types
│   ├── base_trt.py             # TRTModel (ABC) + MonoDepthTRT / StereoDepthTRT
│   ├── base_onnx.py            # ONNXModel (ONNX Runtime session wrapper)
│   └── cam_structure.py        # CameraIntrinsics, CameraExtrinsics, read_calib_file
├── depth_models/
│   ├── a3f/                    # Any2Full — RGB-D depth densification
│   │   ├── any2full.py         # Any2Full_PT2 (torch.export runtime + pre/post)
│   │   └── utils.py            # remove_outliers (sparse-depth denoising)
│   ├── da3/                    # Depth Anything 3
│   │   ├── model/
│   │   │   ├── base_da3.py     # BaseDA3Model mixin (letterbox pre / align / output-map)
│   │   │   ├── da3anyview.py   # DA3AnyViewPT2 (torch.export any-view backend)
│   │   │   ├── da3metric.py    # DA3MetricPT2 (torch.export metric-depth backend)
│   │   │   └── da3nested.py    # DA3NestedPT2 (any-view + metric pipeline + align)
│   │   └── utils/              # alignment.py, geometry.py, pose_align.py
│   ├── streaming/              # chunked multi-camera streaming + alignment
│   │   ├── base_streaming.py   # BaseStreaming (chunking, SIM3 chain, saving)
│   │   ├── da3_streaming.py    # DA3_Streaming backend (two .pt2 checkpoints)
│   │   ├── vggt_omg_streaming.py  # VGGT_OMG_Streaming backend (torch.export)
│   │   ├── configs/base_config.yaml  # alignment lib/method, loop-closure settings
│   │   └── loop_utils/         # sim3utils, alignment_torch/triton, config_utils
│   └── vggt_omega/             # VGGT-Omega
│       ├── base_vggt_omega.py  # BaseVGGTOmega (letterbox, feed assembly, pose/depth post)
│       └── vggt_omega.py       # concrete wrapper + move_lifted_state_to_device
├── det_seg_models/
│   └── sam3/                   # SAM3 promptable segmentation (.pt2 runtime)
├── flow_models/
│   ├── waft/                   # WAFT optical flow — WAFTOnnx (ONNX) / WAFT (TRT)
│   └── tapip3d/                # TAPIP3D 3D tracking (ONNX encoder/updater/corr_forward)
└── utils/
    ├── astribot_dataloader.py  # Astribot dataset: load_rgbd, CAMERA_SETS, depth configs
    ├── image_io.py             # ImageInput, to_image_tensor
    ├── depth_utils.py, streaming_utils.py, video_io.py
    └── visualize_*.py          # depth / flow / mask / tapip3d visualizers + export_glb
tools/
├── export_trt.py               # ONNX → TensorRT engine via trtexec
├── general_test/               # inference + visualization entry points
│   ├── run_stream.py           # streaming CLI (--backend da3|vggt_omega)
│   ├── infer_waft.py, infer_tapip3d.py
│   ├── test_any2full.py, test_sam3.py
│   └── visualize_stream.py, visualize_rgbd.py
└── hifi-umi/                   # HiFi-UMI dataset preprocessing (extract_frames, generate_masks)
scripts/                        # ready-to-run pipeline scripts (infer_*.sh,
                                # visualize_stream.sh, export_trt_docker.sh)
```

Model weights are expected under `weights/` (git-ignored): `any2full/`, `da3/`,
`vggt_omg/`, `sam3/`, `waftv2/`, `tapip3d/`.

## Class hierarchy

Backend bases (in `src/base`) are runtime-agnostic — they own engine/session
load, IO-tensor metadata, input-geometry resolution, and a name-matched
execution helper.

- `TRTModel(ABC)` — TensorRT engine wrapper. Uses the TensorRT 10.3+
  `set_tensor_address` API (no manual CUDA memory management). `_run` binds
  inputs **by name**, allocates outputs, executes on the current CUDA stream.
  `MonoDepthTRT` / `StereoDepthTRT` are legacy single-image / stereo-pair
  helpers.
- `ONNXModel` — ONNX Runtime session wrapper (CUDA EP + CPU fallback); `run(feed)`.

Model-specific wrappers:

- `BaseDA3Model` — pure mixin (needs `self.target_h/target_w` from the
  backend base — no `__init__` of its own). Letterbox preprocessing
  (ImageNet-normalized), extrinsics normalization, any-view feed assembly,
  output-key mapping, mono-sky post-processing, and alignment orchestration
  (`align_with_metric`, `align_to_input`).
- `DA3NestedPT2` (`depth_models/da3/model/da3nested.py`) — the concrete DA3
  runtime: composes two **independent torch.export checkpoints** —
  `DA3AnyViewPT2` (any-view graph, fixed `num_views`, image-only input,
  geometry from the `<N>x<W>x<H>` file-name convention) and `DA3MetricPT2`
  (metric depth; `<W>x<H>` from the file name, or probed via a dummy
  forward) — then aligns any-view depth to metric via `align_with_metric`.
  The metric component can be swapped for any metric `.pt2` sharing the
  any-view resolution without re-exporting the any-view graph. No
  `depth_anything_3` or xFormers dependency.
- `BaseStreaming` — shared streaming pipeline; concrete backends implement two
  hooks: `_load_model()` (sets `self.model_num_views`) and
  `_process_chunk(start, end) -> {"depth", "conf", "extrinsics", "intrinsics"}`.
  Backends: `DA3_Streaming` (any-view + metric `.pt2`), `VGGT_OMG_Streaming`
  (torch.export).
- `Any2Full_PT2` (`depth_models/a3f/any2full.py`) — torch.export runtime with
  `preprocess` / `infer` / `postprocess`; see the Any2Full section below.
- `WAFTOnnx(WAFTBase, ONNXModel)` / `WAFT(WAFTBase, TRTModel)` — optical flow
  (BGR in, flow out); `infer_waft.py --backend trt|onnx`.
- TAPIP3D — ONNX encoder/updater/corr_forward stream runtime under
  `flow_models/tapip3d/`; SAM3 — standalone `torch.export` image runtime
  (fixed 1008 resolution, fixed box-prompt slot count).

## Streaming pipeline (DA3 / VGGT-Omega)

`BaseStreaming.run()` processes synchronized camera folders in sliding chunks
of `chunk_size` frames, aligns consecutive chunks **full-vs-full via SIM3**
(`weighted_align_point_maps` + `accumulate_sim3_transforms`), and saves
per-view `depth` + warped `extrinsics` + `intrinsics` per frame.

- `chunk_size` is derived from the model's fixed `num_views`; a short final
  chunk is padded at its start (copies of the previous chunk's tail) and the
  padding is discarded when saving.
- Optional `mask_dirs` (binary motion masks, >127 = moving) zero the
  confidence of moving pixels **during chunk alignment only** — depth outputs
  are unaffected.
- Alignment library and method come from `configs/base_config.yaml`:
  `align_lib: triton|torch|numba|numpy`, `align_method: sim3|se3|scale+se3`.
  The default triton SIM3 kernel uses atomics and is nondeterministic at the
  ~1e-6 extrinsics level (depth is bit-identical) — see
  [[vggt-stream-alignment-nondeterminism]].
- Optional loop closure (`loop_enable`): a SALAD global-descriptor matcher
  detects revisited scenes and re-aligns via a loop SIM3 optimizer.
- Entry point: `tools/general_test/run_stream.py --backend da3|vggt_omega|a2f`
  (wrappers: `scripts/general_test/infer_stream_stereo.sh` /
  `infer_stream_rgbd.sh`, `scripts/visualize_stream.sh`).

## Any2Full (RGB-D depth densification)

`Any2Full_PT2` loads a `torch.export` program (fixed **480×640** input) and
wraps it deterministically:

1. **Preprocess** — ImageNet-normalized RGB + sparse metric depth (the RGB-D
   sensor's depth, used as the prompt), resized to 480×640 (bilinear / nearest).
   Optional `remove_outliers` denoising of the sparse depth first.
2. **Infer** — the exported graph runs with `init_scaling` disabled and
   returns `(depth, disparity_pre_internal, prompt_depth_resized)`.
3. **Postprocess** — metric scale recovery via an **affine least-squares fit**
   (`torch.pinverse`) of the predicted disparity against the sparse anchors
   (deterministic — the jittered in-graph version is not exported), disparity
   → depth (`1/(d+eps)`), unresize to input resolution, clamp to
   `[min_depth, max_depth]`.

Run: `tools/general_test/test_any2full.py` loads an Astribot head RGB-D frame
(`utils.astribot_dataloader.load_rgbd`), runs the model, and exports a
coloured point cloud (`utils.visualize_depth.export_glb`).

## Key data types (`src/base/cam_structure.py`)

- `CameraIntrinsics` (dataclass, slots) — `fx, fy, cx, cy, bl (baseline,
  metres), w, h`. `from_intrinsics_matrix()`, `from_calib_file()`, per-axis
  scaled intrinsics, `disparity_to_depth()`, `back_project()` /
  `forward_project()`.
- `CameraExtrinsics` (dataclass, slots) — `R (3×3), T (3×1)`;
  `from_calib_file()`, `transform_pointcloud()`.
- `read_calib_file()` parses 3 JSON calibration layouts (see below), returning
  `Intrinsics`, `Extrinsics`, `Resolution`, `Baseline`, `Distortions` keyed by
  camera name.

## Common commands

```bash
# --- Streaming (multi-camera depth + pose) ---
python tools/general_test/run_stream.py --backend vggt_omega \
    --input-dirs <left_dir> <right_dir> \
    --start-frame 210 --max-frames 160 --interval 4 \
    --output-dir output/stream_stereo_vggt_omega
# or the DA3 backend (two .pt2 checkpoints, see
# scripts/general_test/infer_stream_stereo.sh);
# or via the wrapper script:
bash scripts/general_test/infer_stream_stereo.sh

# --- Any2Full RGB-D densification (single frame -> .glb point cloud) ---
python tools/general_test/test_any2full.py \
    --pt2 weights/any2full/Any2Full_vitl_bf16.pt2 --frame_idx 0 --out_dir ./output/a2f

# --- WAFT motion masks / TAPIP3D traces ---
bash scripts/infer_waft.sh
bash scripts/infer_tapip3d.sh

# --- Export ONNX → TRT engine ---
python tools/export_trt.py weights/waftv2/waftv2_dinov3_i5_640x480.onnx
# or inside the NVIDIA TensorRT container (avoids local env setup):
bash scripts/export_trt_docker.sh /abs/path/to/model.onnx fp16
```

## Dataset loading

Inference tools read the **Astribot** dataset via
`utils/astribot_dataloader.py`, which hard-codes `assets/` (the repo's `assets`
symlink points here): `astribot_test_imgs/` (per-camera RGB-D frames) +
`astribot_cam_calib/astribot_calibration_full_640x480.json`. Key API:

- `load_rgbd(frame_idx, camera_name)` → `(rgb_path, depth_metrics_m, ext, ixt)`.
- `CAMERA_SETS`: `set0` (2 views, stereo L/R), `set1` (3 views, head_rgbd +
  stereo L/R), `set2` (4 views, + torso_rgbd).
- `CAMERA_DEPTH_CONFIG` — per-camera depth shape + scale (e.g. head_rgbd
  960×1280 @ 0.001 m/unit; wrists 360×640 @ 0.0001).
- `load_calib()` scales all intrinsics to 640×480 on load; depth is loaded
  from lz4-compressed uint16 files (`load_depth_lz4`).

## TensorRT / backend version requirements

TensorRT code uses the 10.3+ `set_tensor_address` API. Engine files are
version-locked — on a deserialization failure, rebuild with the matching
TensorRT version via `scripts/export_trt_docker.sh`. Pinned runtimes
(pyproject): `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`,
`tensorrt-cu12==11.1.0.106`, `onnxruntime-gpu==1.28.0`. TRT 11 removed the
`--fp16`/`--precisionConstraints`/`--layerPrecisions` trtexec flags (builds
are strongly typed by the ONNX's own dtypes); `tools/export_trt.py` builds
with `--decomposableAttentions='*'` — required for the fused opset-25
Attention nodes in the DA3 ONNX exports (see [[trt11-export-attention-fix]]).
`--precision fp16` pre-casts the ONNX via `cast_onnx_fp16.cast_to_fp16`
(onnxconverter-common, LayerNorm kept fp32); the docker script does the cast
host-side since the tensorrt container has no onnx tooling. fp16 metric
engine: ~6.9 ms vs ~36.9 ms tf32 on an RTX 5090, 644 MB vs 1310 MB.

## Model input/output conventions

- **DA3 preprocessing** (`BaseDA3Model`): BGR → RGB, aspect-preserving resize
  with a 2-decimal-truncated scale, center-pad (letterbox), ImageNet
  normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
  Intrinsics are adjusted for the scale + pad. NCHW; any-view batches to
  `(1, N, 3, H, W)`.
- **DA3 output-name resolution**: outputs are matched by keyword
  (`depth`/`depth_conf`/`pred_extrinsics`/`pred_intrinsics`/`sky`), so exact
  tensor names from the export don't matter. `uses_extrinsics` reads the
  loaded graph's inputs, so whether pose is fed is driven by the model, not a
  caller flag (the shipped any-view export is image-only).
- **Any2Full**: fixed 480×640 input (any size resized internally); depth input
  is the sparse metric depth in metres; output is metric depth (fp32).
- **WAFT**: BGR input by default; motion masks from flow-magnitude threshold
  (pixels >127 are moving).
- **TAPIP3D**: the ONNX updater is exported for a **fixed query count (1088)** —
  an 8×8 bbox grid (64) + 32×32 support grid (1024) matches it exactly.
- **SAM3**: fixed 1008 resolution and a fixed number of box-prompt slots in
  the exported graph (callers right-pad prompts).
- All models use NCHW and GPU tensors; the wrappers handle HWC→CHW, batch dim,
  GPU transfer, and dtype matching the engine's expected input dtype.

## Calibration file formats

`read_calib_file()` handles three calibration JSON layouts:

1. **Full calibration**: `{"camera": {cam_name: {...}}, "lidar": ...}`
2. **Multi-camera**: `{cam_name: {...}}` (top-level keys are camera names)
3. **Single-camera**: flat dict with `resolution`/`intrinsics`/`extrinsics`/
   `distortions` at top level (camera name derived from the filename stem)

Stereo baselines are auto-derived from extrinsics when `parent_frame` names
another known camera (baseline in mm = L2 norm of translation, m→mm).
