# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Multi-model 3D-vision inference, each model deployed on the runtime that fits
it best (ONNX Runtime, TensorRT, or `torch.export` programs):

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
- **TAPIP3D** — 3D point tracking (`torch.export` / `.pt2`), fixed query count.
- **SAM3** — promptable segmentation (`torch.export` / `.pt2`).

Backend-agnostic engine/session wrappers and camera types live in `src/base`;
model-specific pre/post-processing lives under `src/depth_models`,
`src/flow_models`, and `src/det_seg_models`. The repo uses **uv** for
environment management and **hatchling** as the build backend. Core
dependencies: TensorRT, ONNX Runtime, PyTorch, OpenCV, NumPy, trimesh. Requires
**Python ≥3.12.1, <3.13** (TensorRT / ONNX Runtime wheel availability; the
3.12.1 floor excludes 3.12.0, where multiprocess 0.70.19 crashes at exit on
the missing `RLock._recursion_count`). The only automated tests are CPU-only
unit tests in `tests/test_image_io.py`
(`uv run --extra dev pytest tests/ -q`; `dev` is an optional-dependencies
extra, not a PEP-735 group).

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
│   ├── romav2/                 # RoMaV2 cross-image matching — RoMaV2PT2 (.pt2 runtime)
│   ├── rex_omni/               # RexOmni detection wrapper (RexOmniWrapper; .venv-rexomni)
│   └── sam3/                   # SAM3 promptable segmentation (Sam3Image, .pt2 runtime)
├── flow_models/
│   ├── waft/                   # WAFT optical flow — WAFTOnnx (ONNX) / WAFT (TRT)
│   ├── waftv2/                 # WAFTv2_PT2 (torch.export .pt2, bf16) — preprocess/run/postprocess
│   └── tapip3d/                # TAPIP3D 3D tracking (Tapip3D_PT2: .pt2 encoder + fused corr/updater)
└── utils/
    ├── astribot_dataloader.py  # Astribot dataset: load_rgbd, CAMERA_SETS, depth configs
    ├── keyframe_utils.py       # Step-3 key-frame discovery (episode + folder layouts)
    ├── image_io.py             # ImageInput, to_image_tensor, letterbox, imagenet_normalize
    ├── depth_utils.py, streaming_utils.py, video_io.py
    └── visualize_*.py          # depth / flow / mask / tapip3d visualizers + export_glb
tools/
├── export_trt.py               # ONNX → TensorRT engine via trtexec
├── general_test/               # Case 1 (README): general-purpose inference + viz entry points
│   ├── module/                 # one tool per model — infer_<model>.py (test each component)
│   │   ├── infer_any2full.py   # RGB-D depth densification (single frame → .glb)
│   │   ├── infer_waft.py       # dense optical flow → motion masks
│   │   ├── infer_rexomni.py    # open-vocabulary detection (.venv-rexomni)
│   │   ├── infer_sam3.py       # promptable segmentation
│   │   ├── infer_romav2.py     # cross-image keypoint matching
│   │   └── infer_tapip3d.py    # 3D point tracking
│   └── pipeline/               # end-to-end README-step tools
│       ├── run_depth_stream.py           # Step 2: streaming CLI (--backend da3|vggt_omega|a2f)
│       ├── visualize_stream.py     # trajectory video renderer
│       ├── visualize_rgbd.py       # RGB-D frame renderer
│       ├── run_object_detection.py   # Step 3a: RexOmni detections on key-frames (.venv-rexomni)
│       ├── run_object_init_points.py  # Step 3b: SAM3 masks + RoMAv2 init points
│       └── run_e2e_init_points.py         # Step 3 driver on a key-frame folder (3a + 3b)
├── astribot/                   # Case 2 (README): online streaming, no frame extraction
│   ├── extract_frames.py       # sub-task splits, key-frame jpgs, per-subtask videos/frames (+ depth .lz4)
│   ├── run_step2_depth_stream.py   # online per-sub-task depth+pose streaming (da3/vggt_omega/a2f)
│   ├── run_step3_init_points.py    # Step 3 driver on episodes (extract key-frames + 3a + 3b)
│   ├── run_subtask_a3f.py      # online per-sub-task Any2Full RGB-D densification
│   └── visualize_subtask_stream.py  # per-sub-task trajectory videos, online (no re-inference)
├── push_ckpt_2HF.py            # upload weights/ to Hugging Face (Chuong98vt/TraceEngine)
└── hifi-umi/                   # HiFi-UMI dataset preprocessing (extract_frames, generate_masks)
scripts/                        # ready-to-run pipeline scripts
├── general_test/               # wrappers: infer_waft.sh, infer_stream_stereo/rgbd.sh,
│                               #   infer_tapip3d.sh, infer_step3.sh, visualize_stream.sh,
│                               #   export_trt_docker.sh
└── astribot/                   # extract_frames.sh, run_subtask_stream.sh (+ visualize helper)
docs/
├── general_test/               # general_test.md master index (mermaid pipeline diagram) +
│   │                           #   module/ per-model pages (streaming, any2full, waft,
│   │                           #   rexomni, sam3, romav2, tapip3d, visualize_rgbd) +
│   │                           #   pipeline/ step tests (step2, step3)
│   ├── module/                 # one doc per model component
│   └── pipeline/               # end-to-end README-step tests (step2, step3)
└── astribot/                   # extract_frames / subtask-streaming / visualization guides
```

Model weights are expected under `weights/` (git-ignored): `any2full/`, `da3/`,
`vggt_omg/`, `sam3/`, `waftv2/`, `tapip3d/`. They are hosted on Hugging Face
(`Chuong98vt/TraceEngine`): download and symlink with
`hf download Chuong98vt/TraceEngine --local-dir <dir> && ln -s <dir> weights`
(README §2 has the expected layout; `tools/push_ckpt_2HF.py` re-uploads).

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
  (legacy exception: BGR numpy in, flow out — the repo's only non-RGB
  runtime; its own cv2/numpy `_load_image` + `bgr_input`), used for motion
  masks by `tools/astribot/run_step2_depth_stream.py`. `WAFTv2_PT2`
  (`flow_models/waftv2/`, driven by `tools/general_test/module/infer_waft.py`)
  is the torch.export `.pt2` runtime — unified on the shared `image_io` path:
  RGB `ImageInput` + tensor-first trunc2 `letterbox`, bf16 [0,255] feed.
- `Tapip3D_PT2` / `Tapip3DStreamPT2` (`flow_models/tapip3d/`) — TAPIP3D
  torch.export stream runtime: encoder + fused corr/updater iteration
  programs, sliding-window orchestration. SAM3 — standalone `torch.export`
  image runtime (fixed 1008 resolution, fixed box-prompt slot count).

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
- Entry point: `tools/general_test/pipeline/run_depth_stream.py --backend da3|vggt_omega|a2f`
  (wrappers: `scripts/general_test/infer_stream_stereo.sh` /
  `infer_stream_rgbd.sh` / `visualize_stream.sh`). The online per-sub-task
  variant that streams straight from the dataset (no frames on disk) is
  `tools/astribot/run_step2_depth_stream.py` (see
  `docs/astribot/astribot_subtask_depth_stream.md`).

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

Run: `tools/general_test/module/infer_any2full.py` loads an Astribot head RGB-D frame
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
# --- CPU-only unit tests (dev is an optional-dependencies extra, not a PEP-735 group) ---
uv run --extra dev pytest tests/ -q

# --- Streaming (multi-camera depth + pose) ---
python tools/general_test/pipeline/run_depth_stream.py --backend vggt_omega \
    --input-dirs <left_dir> <right_dir> \
    --start-frame 210 --max-frames 160 --interval 4 \
    --output-dir output/stream_stereo_vggt_omega
# or the DA3 backend (two .pt2 checkpoints, see
# scripts/general_test/infer_stream_stereo.sh);
# or via the wrapper script:
bash scripts/general_test/infer_stream_stereo.sh

# --- Any2Full RGB-D densification (single frame -> .glb point cloud) ---
python tools/general_test/module/infer_any2full.py \
    --pt2 weights/any2full/Any2Full_vitl_bf16.pt2 --frame_idx 0 --out_dir ./output/a2f

# --- WAFT motion masks / TAPIP3D traces ---
bash scripts/general_test/infer_waft.sh
bash scripts/general_test/infer_tapip3d.sh

# --- Extract sample frames from a LeRobotDataset (feeds the general_test tools) ---
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 --mode frames \
    --camera-idxes 0 1 --interval 4 --max-frames 50

# --- Online per-sub-task streaming straight from the dataset (no frame extraction) ---
python tools/astribot/run_step2_depth_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 --episode-idxes 0 --backend vggt_omega

# --- Export ONNX → TRT engine ---
python tools/export_trt.py weights/waftv2/waftv2_dinov3_i5_640x480.onnx
# or inside the NVIDIA TensorRT container (avoids local env setup):
bash scripts/general_test/export_trt_docker.sh /abs/path/to/model.onnx tf32
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
TensorRT version via `scripts/general_test/export_trt_docker.sh`. Pinned runtimes
(pyproject): `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`,
`tensorrt-cu12==11.1.0.106`, `onnxruntime-gpu==1.28.0`. TRT 11 removed the
`--fp16`/`--precisionConstraints`/`--layerPrecisions` trtexec flags (builds
are strongly typed by the ONNX's own dtypes); `tools/export_trt.py` builds
with `--decomposableAttentions='*'` — required for the fused opset-25
Attention nodes in the DA3 ONNX exports (see [[trt11-export-attention-fix]]).
Only `tf32` (default, fp32 data with TF32 tensor-core math) and `fp32`
(`--noTF32`) precisions are supported.

## Model input/output conventions

- **Image input contract**: every image-ingesting runtime class accepts the
  full `ImageInput` union — a path, a `PIL.Image`, an HWC numpy array or a
  CHW tensor — via a uniformly named `_load_image(image) -> torch.Tensor`
  (decode-only; CHW **uint8 RGB** on CPU; float `[0,1]` CHW tensors are
  rescaled on entry). Pixel space is **RGB uint8** repo-wide (numpy sources
  must be uint8). Preprocessing is **tensor-first**: decode → letterbox →
  normalize all run as torch ops via the shared `src/utils/image_io.py`
  helpers (`to_image_tensor`, `to_pixel_uint8`, `letterbox`,
  `imagenet_normalize`), so `torch.export` graphs receive tensors straight
  from preprocess — no numpy bounce at the feed boundary. **The
  `flow_models/waft` ONNX/TRT backends are the legacy exception** — BGR +
  numpy + cv2, their own `_load_image` and `bgr_input` — until the unified
  pipeline replaces them; call sites that feed them flip RGB→BGR locally
  (e.g. `tools/astribot/run_step2_depth_stream.py`'s motion-mask path).
  `WAFTv2_PT2` (`flow_models/waftv2/`) follows the unified path: shared
  `_load_image` decode to CHW uint8 RGB, shared trunc2 `letterbox`, feed in
  [0, 255] (the exported graph normalizes internally).
- **DA3 preprocessing** (`BaseDA3Model`): shared `_load_image`, then
  `letterbox(scale_mode="trunc2")` — aspect-preserving resize with a
  2-decimal-truncated uniform scale, center-pad — and `imagenet_normalize`
  (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
  `preprocess_views(imgs: list[ImageInput])` batches N views to
  `(1, N, 3, H, W)` fp32 CPU. Intrinsics are adjusted for the scale + pad.
- **VGGT-Omega** (`BaseVGGTOmega.infer(images: list[ImageInput])`): shared
  `_load_image`, then `letterbox(scale_mode="round", bicubic, antialias)` —
  the raw-scale VGGT convention — followed by float `[0,1]` canvas assembly.
- **DA3 output-name resolution**: outputs are matched by keyword
  (`depth`/`depth_conf`/`pred_extrinsics`/`pred_intrinsics`/`sky`), so exact
  tensor names from the export don't matter. `uses_extrinsics` reads the
  loaded graph's inputs, so whether pose is fed is driven by the model, not a
  caller flag (the shipped any-view export is image-only).
- **Any2Full**: `preprocess(rgb: ImageInput, depth_metrics)` — the RGB leg is
  decoded via `_load_image` and resized to the fixed 480×640 input (any size
  resized internally); depth input is the sparse metric depth in metres;
  output is metric depth (fp32).
- **RoMaV2** (`RoMaV2PT2`): `_load_image` is decode-only; the float/255,
  batching and device moves happen in `match_pair`.
- **WAFT** (`flow_models/waft`, ONNX/TRT): BGR input by default (legacy
  exception, see above); motion masks from flow-magnitude threshold (pixels
  >127 are moving). **WAFTv2** (`WAFTv2_PT2`, `.pt2`): unified RGB
  `ImageInput`, bf16 [0,255] feed to a graph that normalizes internally.
- **TAPIP3D**: the shared frame loader `utils.streaming_utils.load_batch_frames`
  decodes via `image_io` into `[T, 3, H, W]` uint8; the `.pt2` iteration
  program is exported for a **fixed query count (1088)** — an 8×8 bbox grid
  (64) + 32×32 support grid (1024) matches it exactly (SAM3 mask-sampled
  queries fall back to the grid when the mask is too small).
- **SAM3**: `preprocess_image` decodes via the shared `_load_image`; fixed
  1008 resolution and a fixed number of box-prompt slots in the exported
  graph (callers right-pad prompts).
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
