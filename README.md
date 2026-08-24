# 3D Trace Estimation Engine

This repo provides an end-to-end pipeline that turns **one or several folders
of frames extracted from synchronized camera videos** into the **3D trace of a
moving object**, along with camera poses and metric depth for the whole
sequence. It combines several models, each deployed with the runtime it fits
best (ONNX Runtime, TensorRT, or TorchScript / `torch.export`):

- **WAFT** — dense optical flow, used to build motion masks.
- **DA3 / VGGT-Omega (streaming)** — multi-camera depth + camera pose estimation
  over long trajectories.
- **TAPIP3D** — 3D point tracking (traces).

## Pipeline

```mermaid
flowchart LR
    A[Sync camera frames] --> B[WAFT motion masks]
    B --> C[Streaming pose + depth]
    C --> D[Visualization]
    C --> E[TAPIP3D 3D traces]
```

1. **Input** — one or several folders of synchronized frames (one per camera),
   e.g. `demo_data/astribot_stereo_lrb/extract_frames/{stereo_left,stereo_right}`.
2. **Motion masks** — `scripts/infer_waft.sh` runs WAFT optical flow and
   thresholds the flow magnitude into per-pixel motion masks. The masks mark
   moving pixels; the next step uses them to **select static points for aligning
   the cameras over a long trajectory**.
3. **Camera pose + depth** — `scripts/infer_stream.sh` runs the streaming
   multi-camera model (VGGT-Omega or DA3) over the frames in sliding chunks and
   aligns the chunks into one consistent world frame, producing per-frame metric
   depth and camera poses for every view. `scripts/visualize_stream.sh` renders
   the depth and camera motion as a video.
4. **3D traces** — `scripts/infer_tapip3d.sh` runs TAPIP3D on top of the pose +
   depth output and tracks the object over time. You need to provide the
   **bounding box** of the tracked object (e.g. derived from a segmentation
   mask).

Each step is detailed below with its reference script and key flags.

## Installation

Install the environment simply by:

```bash
uv sync
```

with pinned inference runtimes: `torch==2.7.1+cu128`,
`onnxruntime-gpu==1.28.0`, `tensorrt-cu12==11.1.0.106`.

## Layout

```
data                 # Astribot sample data (symlink)
demo_data            # demo sequences (symlink)
src/
├── base/            # TRTModel / ONNXModel wrappers, CameraIntrinsics/Extrinsics
├── depth_models/
│   ├── da3/         # Depth Anything 3 (metric / any-view / nested)
│   ├── vggt_omega/  # VGGT-Omega (torch.export program)
│   └── streaming/   # chunked multi-camera streaming + long-trajectory alignment
├── flow_models/
│   ├── waft/        # WAFT optical flow (ONNX + TRT)
│   └── tapip3d/     # TAPIP3D 3D point tracking (ONNX)
└── utils/           # dataloaders, visualization, streaming helpers
tools/               # inference + visualization entry points
scripts/             # ready-to-run pipeline scripts (infer_waft, infer_stream,
                     # visualize_stream, infer_tapip3d, export_trt_docker)
weights/             # model weights (ONNX / TRT / TorchScript), clone from HG
```

## Model weights

The model weights are expected under `weights/` in the following structure:

```
weights
├── vggt_omg/
│   └── vggt_omg_24x640x480.pt2
├── da3/
│   ├── da3_anyview_24x644x490_giant-large-1.1.pt        # TorchScript (streaming)
│   ├── da3_anyview_24x644x490_giant-large-1.1.pt2
│   └── ...                                              # metric/any-view ONNX + TRT
├── waftv2/
│   ├── waftv2_dinov3_i5_640x480.onnx
│   └── waftv2_dinov3_i5_640x480_tf32.engine
├── tapip3d/
│   ├── tapip3d_encoder_480x640.onnx
│   ├── tapip3d_updater.onnx
│   └── tapip3d_corr_forward.onnx
└── s2m2/
    ├── S2M2_XL_640_480_v2_torch21.onnx
    └── S2M2_XL_640_480_v2_torch21.engine
```

### Precompiled weights

For convenience, ONNX checkpoints can be downloaded from Hugging Face
[Chuong98vt/DepthModels](https://huggingface.co/Chuong98vt/DepthModels/tree/main). Model name conventions:

- **VGGT-Omega (streaming)**: `vggt_omg_24x640x480` — fixed 24 views per chunk,
  inference at 640×480.
- **DA3 (streaming)**: `da3_anyview_24x644x490` — fixed 24 views per chunk,
  644×490 (must be divisible by 14).
- **WAFT**: `waftv2_dinov3_i5_640x480` — optical flow with DINOv3-small
  backbone, 5 iterative refinements, inferred at 640×480.
- **TAPIP3D**: `tapip3d_encoder_480x640` / `tapip3d_updater` / `tapip3d_corr_forward`
  — the updater is exported for a **fixed query count (1088)**.

(Optional) To export the ONNX models yourself, clone the PyTorch source repos
and run their `export_onnx.py`:
- DA3: https://github.com/Chuong98CC/Depth-Anything-3
- S2M2: https://github.com/Chuong98CC/s2m2
- WAFT: https://github.com/Chuong98CC/WAFT
- TAPIP3D / VGGT-Omega: see the model sources for the export scripts.

### Export ONNX → TensorRT

Export a TRT engine using Docker simply by:

```bash
bash scripts/export_trt_docker.sh /abs/path/to/model.onnx fp16 (or fp32)
```

- For the WAFT model, the precision `fp32` is required.
- Other models can be exported with `fp16` for lower memory inference.

## Step-by-step details

The steps below mirror the demo scripts in `scripts/` (data paths are the
`demo_data/astribot_stereo_lrb` sequence, a two-camera stereo rig).

### 1. Input data

Prepare one folder of extracted frames per synchronized camera, with matching
frame stems across folders:

```
demo_data/astribot_stereo_lrb/extract_frames/stereo_left/frame_*.jpg
demo_data/astribot_stereo_lrb/extract_frames/stereo_right/frame_*.jpg
```

### 2. Motion masks (WAFT optical flow)

`scripts/infer_waft.sh` runs WAFT on a frame folder and thresholds the flow
magnitude into binary motion masks:

```bash
python tools/infer_waft.py --input <frames_dir> --backend trt \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine \
    --start 210 --stride 4 -thr 2 -o mask \
    --output-dir demo_data/astribot_stereo_lrb/motion_mask/
```

- `--stride 4` pairs frames 4 apart; `-thr 2` is the flow-magnitude (pixel
  displacement) threshold above which a pixel counts as moving.
- Output: per-frame masks under
  `<output-dir>/<video_name>/mask/` (e.g. `stereo_left/mask/`). Pixels above
  127 are moving.
- Run once per camera; the masks are consumed by the streaming step, where
  moving-pixel confidence is zeroed so that **only static points drive the
  long-trajectory camera alignment**.

### 3. Camera pose and depth (streaming)

`scripts/infer_stream.sh` runs the streaming multi-camera model over all camera
folders in sliding chunks and aligns the chunks into a single world frame:

```bash
python tools/run_stream.py \
    --backend vggt_omega \            # or "da3"
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --mask-dirs <left_mask_dir> <right_mask_dir> \
    --chunk-size 24 \
    --start-frame 210 --max-frames 160 --interval 4 \
    --output-dir output/masked_stream_stereo_vggt_omega
```

- `--backend` selects the pre-exported model: `vggt_omega` (VGGT-Omega
  `torch.export` program, `weights/vggt_omg/vggt_omg_24x640x480.pt2`) or `da3`
  (TorchScript any-view). Both exports have a **fixed number of views**, so
  `--chunk-size` must equal 24 and be divisible by the number of cameras.
- `--mask-dirs` (optional) supplies the motion masks from step 2, used during
  chunk alignment only — outputs are unaffected.
- `--interval 4` / `--max-frames` subsample the sequence (every 4th frame).
- Output: per camera folder `depth_<camera_name>/frame_<idx>.npz` containing
  `depth` (H, W, metric), `extrinsics` (3×4, world→camera), `intrinsics`
  (3×3), plus `timings.json`. A short final chunk is padded automatically.
- Add `--video` to render the trajectory video right after the run.

**Visualization** — `scripts/visualize_stream.sh` renders the saved results as
a video of the coloured depth point cloud with camera frustums and the growing
camera path:

```bash
python tools/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/masked_stream_stereo_vggt_omega \
    --output output/masked_stream_stereo_vggt_omega/vggt_omega_stream.mp4 \
    --fps 30 --size 960x540
```

### 4. 3D traces (TAPIP3D)

`scripts/infer_tapip3d.sh` tracks the object in 3D over the sequence, using the
frames plus per-frame geometry (depth + camera poses) of the view the object is
tracked in:

```bash
python tools/infer_tapip3d.py \
    --encoder weights/tapip3d/tapip3d_encoder_480x640.onnx \
    --updater weights/tapip3d/tapip3d_updater.onnx \
    --corr_forward weights/tapip3d/tapip3d_corr_forward.onnx \
    --image_dir <frames_dir> \
    --npz_dir <geometry_dir> \
    --output_dir output/tapip3d_stereo_left_onnx \
    --start_frame 210 --fps 4 \
    --bbox 200 265 260 300 \          # x0 y0 x1 y1 of the tracked object
    --grid_x 8 --grid_y 8 --support_grid_size 32 \
    --num_iters 6 --vis_threshold 0.5 --visualize
```

- `--bbox x0 y0 x1 y1` selects the object to track (derive it from a
  segmentation mask if you have one). A grid of points inside the bbox is
  unprojected to world-space queries using the first frame's depth and pose.
- `--npz_dir` is the geometry folder: per-frame `frame_<idx>.npz` with
  `depth` (H, W), `extrinsic` (4×4, world→camera) and `intrinsic` (3×3) — i.e.
  the pose + depth output of step 3 (the demo sequence ships with a
  pre-generated copy under `demo_data/astribot_stereo_lrb/geometry/`).
- The shipped updater ONNX has a **fixed query count (1088)**: the default
  8×8 bbox grid (64) + 32×32 support grid (1024) matches it exactly; the
  script exits with a clear error otherwise (e.g. depth holes in the bbox drop
  points).
- `--fps 4` subsamples the sequence (inference runs at 4 Hz on the 30 Hz data).
- Output: `coords.npy` (T, Q, 3) — the world-space 3D traces, `visibs.npy`
  (T, Q) — per-point visibility, and `metadata.json`. With `--visualize` a
  rendering of the tracks is saved as a video.

## Notes

See the per-model docs in [`docs/`](docs/) for model-specific usage notes
(preprocessing, output formats, conventions).
