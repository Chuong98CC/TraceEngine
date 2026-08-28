# 3D Trace Estimation Engine

This repo provides an end-to-end pipeline that extract the **3D trace of a
moving object**, along with object segmentation mask, camera poses and metric depth for the whole
sequence. It combines several models, each deployed with the runtime it fits
best (ONNX Runtime, TensorRT, or `torch.export` programs):

- **Any2Full** — RGB-D depth densification: grounds a Depth Anything prediction
  with the RGB-D sensor's sparse metric depth as a prompt, producing a densified
  metric point cloud.
- **WAFT** — dense optical flow, used to build motion masks.
- **DA3 / VGGT-Omega (streaming)** — multi-camera depth + camera pose estimation
  over long trajectories.
- **TAPIP3D** — 3D point tracking (traces).
- **SAM3** — promptable segmentation.

## Installation

Install the environment simply by:

```bash
uv sync
```

with pinned inference runtimes: `torch==2.11.0+cu128`,
`onnxruntime-gpu==1.28.0`, `tensorrt-cu12==11.1.0.106`.


## Model weights

The model weights are hosted on Hugging Face at
[`Chuong98vt/TraceEngine`](https://huggingface.co/Chuong98vt/TraceEngine).
Download them and point `weights/` at the download folder:

```bash
pip install -U "huggingface_hub[cli]"
hf download Chuong98vt/TraceEngine --local-dir <download_weight_folder>
ln -s <download_weight_folder> weights
```

> If a `weights/` directory already exists (e.g. an empty placeholder), remove
> it first — `ln -s <download_weight_folder> weights` would otherwise create
> the symlink *inside* the existing directory.

Expected layout (mirrors the repo layout):

```
weights
├── any2full/
│   └── Any2Full_vitl_bf16.pt2            # RGB-D densification, vit-large, bf16
├── vggt_omg/
│   └── vggt_omg_64x640x480_bf16.pt2      # VGGT-Omega streaming, fixed 64 views
├── da3/
│   ├── da3_anyview_64x644x490_giant-large-1.1_bf16.pt2   # any-view, fixed 64 views
│   └── da3_metric_644x490_giant-large-1.1_bf16.pt2       # metric depth
├── sam3/
│   └── sam3_image_exported_bf16.pt2
├── waftv2/
│   └── waftv2_dinov3_i5_640x480.onnx     # ONNX; TRT engine built from it via tools/export_trt.py
└── tapip3d/
    ├── tapip3d_encoder_480x640_bf16.pt2
    └── tapip3d_iteration_1088_bf16.pt2   # fixed 1088-query iteration model
```

## How the code works

The `tools/` folder is the working entry point. Depending on the dataset
size, use one of two tool sets:

### Case 1 — General purpose: test each component individually (`tools/general_test`)

The `tools/general_test` scripts run **one model at a time** end-to-end (load
weights → preprocess → infer → postprocess → save outputs/visualization) on a
**folder of input images** (or a single example image), independent of any
dataset. This is where each individual component is tested and verified:
WAFT optical flow, SAM3 segmentation, Any2Full depth densification,
DA3 / VGGT-Omega streaming, TAPIP3D tracking. Per-tool usage is documented in
[`docs/general_test/general_test.md`](docs/general_test/general_test.md).

To get sample image folders, first extract frames from a LeRobotDataset —
the sample dataset is
[`Kronze157/astri_making_coffee_vlva`](https://huggingface.co/datasets/Kronze157/astri_making_coffee_vlva)
on Hugging Face:

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root <dataset_dir> \
    --mode frames --camera-idxes 0 1 --interval 4 --max-frames 50
```

The extracted per-sub-task jpg folders (each camera paired with its raw
uint16 depth `.lz4`) are then passed as `--input-dirs` to the general_test
tools, e.g.:

```bash
python tools/general_test/run_stream.py --backend vggt_omega \
    --input-dirs <subtask_XX>/<cam0> <subtask_XX>/<cam1> \
    --output-dir output/stream_vggt_omega
```

### Case 2 — Large datasets: online / streaming from the dataset (`tools/astribot`)

For large datasets, extracting every frame to jpg is impractical. The
`tools/astribot` scripts instead stream **online, directly from the
LeRobotDataset** — frames are decoded one chunk at a time and never written
to disk, so no extracted frames or per-subtask videos are needed. Sub-task
splits are taken from the ground-truth `subtask_index` column (gripper-based
inference fallback).

- `run_subtask_stream.py` — per-sub-task depth + pose streaming (DA3 /
  VGGT-Omega / Any2Full backends, optional per-chunk WAFT motion masks).
- `run_subtask_a3f.py` — per-sub-task Any2Full RGB-D densification (RGB-D
  cameras only).
- `visualize_subtask_stream.py` — trajectory video per sub-task, rendered
  online from the saved pipeline outputs (no re-inference).
- `extract_frames.py` — shared dataset introspection + sub-task splitting
  (also the frame-extraction script of case 1).

```bash
python tools/astribot/run_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root <dataset_dir> --episode-idxes 0 --backend vggt_omega
```

The docs live in [`docs/astribot/`](docs/astribot/):
[`astribot_extract_frames.md`](docs/astribot/astribot_extract_frames.md),
[`astribot_subtask_stream.md`](docs/astribot/astribot_subtask_stream.md),
[`astribot_visualize_subtask_stream.md`](docs/astribot/astribot_visualize_subtask_stream.md).
