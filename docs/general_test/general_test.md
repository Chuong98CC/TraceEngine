# General-Test Tools (`tools/general_test`)

Inference + visualization entry points for **testing and verifying each
model's performance** — the "Case 1" tool set of the repo README. The scripts
in this folder are general-purpose: they run on a **folder of images**.

To get sample image folders, first extract frames from a LeRobotDataset with
`tools/astribot/extract_frames.py` (sample dataset
[`Kronze157/astri_making_coffee_vlva`](https://huggingface.co/datasets/Kronze157/astri_making_coffee_vlva)).
Model weights are expected under `weights/` (see the repo README §2 for the
Hugging Face download + symlink). All scripts can be run from the repo root
after `uv sync`.

## Pipeline overview

```mermaid
flowchart LR
    subgraph Step1["Step 1 · Extract frames"]
        DS["LeRobotDataset<br/>(Kronze157/astri_making_coffee_vlva)"] --> EF["tools/astribot/extract_frames.py<br/>--mode frames"]
        EF --> FR["Per-camera frame folders<br/>frame_*.jpg + raw depth .lz4"]
    end

    subgraph Step2["Step 2 · Camera depth + pose estimation"]
        FR --> RS{"run_stream.py<br/>--backend"}
        WF["infer_waft.py — motion masks<br/>(optional)"] --> RS
        RS -->|"vggt_omega / da3"| ST["Stereo / mono<br/>depth + pose"]
        RS -->|"a2f (RGB-D)"| A2F["Any2Full densified<br/>depth + pose"]
        ST --> OUT["depth_&lt;cam&gt;/frame_*.lz4 + .npz<br/>metric depth + camera pose"]
        A2F --> OUT
    end

    subgraph Step3["Step 3 · 3D traces"]
        OUT --> TP["infer_tapip3d.py<br/>3D point tracking"]
        SG["test_sam3.py — object box<br/>+ text prompt"] --> TP
        TP --> TR["coords.npy — 3D traces + visibs.npy<br/>tracks.mp4"]
    end
```

1. **Extract frames** — `tools/astribot/extract_frames.py` turns the
   LeRobotDataset into synchronized per-camera frame folders (paired with raw
   uint16 depth `.lz4` where the camera has a depth feature).
2. **Camera depth + pose estimation** — `run_stream.py` streams the frame
   folders through a depth+pose backend in sliding chunks aligned into one
   world frame: `vggt_omega` / `da3` (mono/stereo) or `a2f` (RGB-D, Any2Full
   densification). Optional WAFT motion masks (`infer_waft.py`) zero the
   confidence of moving pixels during chunk alignment.
3. **3D traces** — `infer_tapip3d.py` tracks world-space points from the
   depth + pose output, anchored in the tracked object's box (segmented by
   SAM3, `test_sam3.py`) on the first frame.

## Tool index

| # | Tool | What it does | Runtime | Doc | Wrapper script |
|---|---|---|---|---|---|
| 1 | `visualize_rgbd.py` | RGB-D visualization (image / depth-lz4 + pose-npz folder pair) | — (rendering) | [`visualize_rgbd.md`](visualize_rgbd.md) | — |
| 2 | `infer_waft.py` | dense optical flow → motion masks | WAFT (ONNX / TensorRT) | [`waft.md`](waft.md) | `scripts/general_test/infer_waft.sh` |
| 3 | `test_sam3.py` | promptable segmentation (box + text) | SAM3 (`.pt2`) | [`sam3.md`](sam3.md) | — |
| 4 | `test_any2full.py` | RGB-D depth densification | Any2Full (`.pt2`) | [`any2full.md`](any2full.md) | — |
| 5 | `run_stream.py` / `visualize_stream.py` | streaming multi-camera depth + pose (incl. RGB-D `a2f`), trajectory video | DA3 / VGGT-Omega / Any2Full (`.pt2`) | [`streaming.md`](streaming.md) (+ [`visualize_stream.md`](visualize_stream.md) for the renderer) | `scripts/general_test/infer_stream_stereo.sh`, `infer_stream_rgbd.sh`, `visualize_stream.sh` |
| 6 | `infer_tapip3d.py` | 3D point tracking over long videos | TAPIP3D (`.pt2`) | [`tapip3d.md`](tapip3d.md) | `scripts/general_test/infer_tapip3d.sh` |

## Ready-to-run scripts (`scripts/general_test/`)

Thin wrappers around the tools above, hard-coded for the Astribot sample
extraction:

| Script | What it runs |
|---|---|
| `infer_waft.sh` | WAFT motion masks on a demo frame folder (TRT engine, `-thr 2`) |
| `infer_stream_stereo.sh` | Stereo streaming (`da3` / `vggt_omega`) on the sample extraction, then renders the trajectory video |
| `infer_stream_rgbd.sh` | RGB-D streaming (`a2f` backend) on the sample extraction, then renders the trajectory video |
| `infer_tapip3d.sh` | 3D point tracking of the coffee cup (`--bbox 1 240 100 340 --text_prompt "brown coffee cup"`), env-configurable via `IMG_DIR` / `DEPTH_DIR` / `OUTPUT_DIR` |
| `visualize_stream.sh` | Trajectory video from a streaming result |
| `export_trt_docker.sh` | Build a TensorRT engine from an ONNX checkpoint inside the NVIDIA TensorRT container (`<onnx_path> [fp16|tf32|fp32]`) |

## Relation to dataset-specific tools

The scripts in this folder are the **general-purpose reference entry
points**. Dataset-specific tool folders reuse their functions but optimize
for a particular dataset:

- `tools/astribot` — Astribot dataset helpers: `extract_frames.py` (the
  sample-frame producer above) and the online streaming tools that never
  write frames to disk (see [`docs/astribot/`](../astribot/)). The
  `tools/astribot` streamers reuse the same streaming backend classes and
  the same output contract, so the same visualizers load their results.
- `tools/hifi-umi` — HiFi-UMI dataset preprocessing (`extract_frames.py`,
  `generate_masks.py`), reusing the same model entry points.

See `docs/astribot/` for the "Case 2" online-streaming workflow of the repo
README.
