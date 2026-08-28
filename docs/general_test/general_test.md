# General-Test Tools (`tools/general_test`)

Inference + visualization entry points for **testing and verifying each
model's performance**. The scripts in this folder are general-purpose: they
run on a **folder of images** or a **single example image** and exercise one
model end-to-end (load weights → preprocess → infer → postprocess → save
outputs / visualization), independent of any specific dataset.

Dataset-specific tool folders (`tools/astribot`, `tools/hifi-umi`) **reuse the
functions from `general_test`** but optimize them for a particular dataset
(naming conventions, camera sets, batching). The ready-to-run pipeline
scripts under `scripts/` are thin wrappers around these entry points.

| # | Script | Model / capability | Runtime |
|---|---|---|---|
| 1 | `visualize_rgbd.py` | RGB-D visualization (image / depth-lz4 + pose-npz folder pair) | — (rendering) |
| 2 | `infer_waft.py` | dense optical flow | WAFT (ONNX / TensorRT) |
| 3 | `test_sam3.py` | promptable segmentation (box + text) | SAM3 (`.pt2`) |
| 4 | `test_any2full.py` | RGB-D depth densification | Any2Full (`.pt2`) |
| 5 | `run_stream.py` / `visualize_stream.py` | streaming multi-camera depth + pose (incl. RGB-D `a2f`), trajectory video | DA3 / VGGT-Omega / Any2Full (`.pt2`) |
| 6 | `infer_tapip3d.py` | 3D point tracking over long videos | TAPIP3D (ONNX) |

Model weights are expected under `weights/` (see the repo README for the
layout). All scripts can be run from the repo root after `uv sync`.

---

<details>
<summary><strong>1. RGB-D visualization — `visualize_rgbd.py`</strong></summary>

Visualizes RGB-D data from **one of two mutually exclusive sources**:

1. **Camera mode** (`--camera_name`) — an Astribot camera frame loaded via
   `utils.astribot_dataloader.load_rgbd` (e.g. `head_rgbd`); metric depth is
   recovered from the greyscale depth image (clip at `--max_depth_m`).
   Sample data are in `assets/astribot_test_imgs`.
2. **Folder mode** (`--rgb_dir` + `--depth_npz_dir`) — pairs RGB images
   (`<stem>.jpg/.jpeg/.png`) by stem with a depth `<stem>.lz4` (raw uint16 mm,
   see `utils.astribot_dataloader.load_depth_lz4`, reshaped to the RGB frame
   size) and a pose `<stem>.npz` (`extrinsics` 3×4/4×4, `intrinsics` 3×3) —
   the format written by `run_stream.py`. `--frame_index` selects the pair at
   that position in the sorted matching stems.

```bash
# Astribot camera frame
python tools/general_test/visualize_rgbd.py --camera_name head_rgbd \
    --frame_index 0 --save_viz --save_glb

# RGB + depth-lz4 folder pair (e.g. a run_stream.py output)
python tools/general_test/visualize_rgbd.py \
    --rgb_dir path/to/rgb --depth_npz_dir path/to/depth \
    --save_viz --save_glb
```

| Argument | Default | Description |
|---|---|---|
| `--camera_name` | — | Astribot camera (mutually exclusive with the folder pair) |
| `--rgb_dir` / `--depth_npz_dir` | — | Folder pair: RGB images + depth `.lz4` / pose `.npz` (mutually exclusive with `--camera_name`) |
| `--frame_index` | `0` | Frame to process (camera mode: index; folder mode: position in sorted stems) |
| `--max_depth_m` | `5.0` | Far-plane clip for recovering metric depth from the greyscale image |
| `--output`, `-o` | `output/rgbd` | Output directory |
| `--save_viz` / `--save_glb` | off | Save the heatmap depth image or save `.glb` point cloud |

</details>

<details>
<summary><strong>2. Optical flow (WAFT) — `infer_waft.py`</strong></summary>

Dense optical flow on a **video file or a folder of extracted frames**, with
ONNX or TensorRT backends. Folder input is auto-detected when `--input` is a
directory: frames are scanned as `frame_{idx}.jpg` / `frame_{idx}.png`,
sorted by index, and per-frame outputs are named after the anchor frame's
source index. Backend is inferred from the checkpoint file extension
(`.onnx` → ONNX Runtime, `.engine` → TensorRT); pass `--backend` when the
path has no extension.

```bash
# Folder of extracted frames, TensorRT, motion masks (pixels > 2 px moving)
python tools/general_test/infer_waft.py --input <frames_dir> --backend trt \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine \
    --start 210 --stride 4 -thr 2 -o mask \
    --output-dir demo_data/astribot_stereo_lrb/motion_mask/

# Video file, ONNX, flow colour video + raw .flo files
python tools/general_test/infer_waft.py --input video.mp4 \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480.onnx \
    --output-mode all
```

| Argument | Default | Description |
|---|---|---|
| `--input` | **required** | Video file or folder of `frame_{idx}.jpg/.png` |
| `--checkpoint` | `weights/waftv2/waftv2_dinov3_i5_640x480` | Model checkpoint; backend inferred from `.onnx` / `.engine` extension |
| `--backend` | auto | `onnx` or `trt`; required only when the checkpoint has no extension |
| `--output-dir` | `./output` | Root output directory (per-input subfolder: `<output-dir>/<video_name>/`) |
| `--output-mode`, `-o` | `flow` | `flow` (colour video), `raw` (`.flo` files), `overlay` (frame+flow overlay), `mask` (motion-mask frames), `flow-mask` (motion-mask video), `all` |
| `--start` | `0` | Frame **index** in folder mode, frames to skip in video mode |
| `--stride` | `1` | Gap between paired frames (≥ 1) |
| `--max-frames` | until EOF | Max number of frame **pairs** to process |
| `--motion-threshold`, `-thr` | — | Pixel-displacement threshold for the binary moving-pixel mask (required for `mask` / `flow-mask`) |
| `--device` | `cuda` | ONNX Runtime device (`cuda` / `cpu`); ignored for TRT |
| `--no-bgr-input` | off | Images are already RGB (skip internal BGR→RGB conversion) |

**Output** — under `<output-dir>/<video_name>/`: `flow/flow.mp4` (colour
wheel), `raw/frame_*.flo` (Middlebury format), `overlay/overlay.mp4`,
`mask/mask_<idx>.jpg` (255 = moving, >127 is moving in the binary masks
consumed by the streaming step) and `mask/mask.mp4`.

Wrapper: `scripts/infer_waft.sh`.

</details>

<details>
<summary><strong>3. Promptable segmentation (SAM3) — `test_sam3.py`</strong></summary>

Runs the exported SAM3 image model (`sam3_image_exported_bf16.pt2`) on box
and text prompts. This script is a **notebook-style example** (`#%%` cells,
editable in any Python/Jupyter environment) rather than a configurable CLI:
the image path and prompt boxes are hard-coded at the top, and the model is
loaded with hard-coded relative paths (`../../weights/...`, run from
`tools/general_test/`).

```python
from det_seg_models.sam3 import Sam3Image

model = Sam3Image("../../weights/sam3/sam3_image_exported_bf16.pt2")

# Box prompts (xywh) + labels; predict() handles pre/infer/post end-to-end
inference_state = model.predict(image, text_prompt="visual",
                                boxes=norm_boxes_cxcywh, labels=[True, True])

# Text-only prompt
text_state = model.predict(image, text_prompt="brown cup")
```

Key points:

- `predict()` returns a state dict with `masks`, `boxes`, `scores`.
- Box prompts are normalized to cxcywh (`box_xywh_to_cxcywh` +
  `normalize_bbox`); `draw_box_on_image` / `plot_results` visualize.
- The graph is exported at a fixed 1008 resolution with a fixed number of
  box-prompt slots; callers right-pad prompts.

</details>

<details>
<summary><strong>4. RGB-D depth densification (Any2Full) — `test_any2full.py`</strong></summary>

Single-frame RGB-D densification on an Astribot frame: uses the RGB-D
sensor's sparse metric depth as a *prompt* to ground a Depth Anything
prediction, recovering the metric scale with a deterministic affine fit, and
exports a coloured point cloud. Runs the `.pt2` runtime
(`Any2Full_PT2`: preprocess → infer → postprocess). Inputs are resized to
the exported fixed 480×640, so any RGB/depth size is accepted; output depth
and the point cloud are at the exported resolution.

```bash
python tools/general_test/test_any2full.py \
    --pt2 weights/any2full/Any2Full_vitl_bf16.pt2 \
    --frame_idx 0 \
    --out_dir ./output/a2f
```

| Argument | Default | Description |
|---|---|---|
| `--pt2` | `weights/any2full/Any2Full_vitl_bf16.pt2` | `torch.export` checkpoint |
| `--camera_name` | `head_rgbd` | Astribot camera (`head_rgbd`, `torso_rgbd`, wrists); frames loaded via `utils.astribot_dataloader.load_rgbd` (metric depth in metres + calibrated intrinsics/extrinsics) |
| `--frame_idx` | `0` | Frame index to load |
| `--out_dir` | `./output/a2f` | Output directory |
| `--denoise` / `--denoise_threshold` / `--denoise_kernel_size` / `--denoise_min_valid` | off / `2.0` / auto / `5` | Denoise the sparse depth before inference (drops isolated/anomalous points) |
| `--init_scaling` | `True` | Enable the deterministic affine scale recovery against the sparse anchors; off = raw graph depth |
| `--max_depth` / `--min_depth` | `10` / `0` | Clamp the final output (metres) |

**Output** — `frame_<idx>.glb` (coloured point cloud back-projected with the
camera intrinsics), `frame_<idx>.npy` (metric depth), `<stem>.png`
(colour-mapped depth).

</details>

<details>
<summary><strong>5. Streaming depth + pose — `run_stream.py` & `visualize_stream.py`</strong></summary>

### run_stream.py

Multi-camera depth + camera pose over a long trajectory, in sliding chunks
aligned into one world frame. Unified entry point for three backends:

- `da3` — two pre-exported DA3 `torch.export` programs: the any-view graph
  (fixed `num_views`) + the metric-depth graph, composed by `DA3NestedPT2`.
  Default weights are overridden with `--anyview-model-path` /
  `--metric-model-path`.
- `vggt_omega` — a single pre-exported VGGT-Omega `torch.export` program
  (fixed `num_views`), overridden with `--model-path`.
- `a2f` — RGB-D: the DA3 any-view graph + Any2Full (RGB + raw sensor depth →
  densified depth, `A2F_NestedPT2`). `--input-dirs` are the RGB camera
  folders; each has a parallel raw-depth folder in `--depth-dirs` (`.lz4`
  uint16 mm maps, loaded via `load_depth_lz4`, scaled to metres by
  `--depth-scale`, 0 = invalid). Any2Full densifies the sparse sensor depth
  and the any-view depth is aligned to it. Only the RGB folders emit output
  (`depth_<name>/`) — the depth folders are inputs only. Default weights are
  overridden with `--anyview-model-path` / `--a2f-model-path`; with
  `--no-depth-enhance` Any2Full is not loaded and the raw sensor depth feeds
  the alignment directly.

`chunk_size` is derived from the model's fixed `num_views` (read from the
export itself) and must be divisible by the number of input folders. A short
final chunk is padded at its start with images from the previous chunk; the
padded outputs are discarded.

```bash
# Stereo (VGGT-Omega or DA3)
python tools/general_test/run_stream.py \
    --backend vggt_omega \                 # or "da3"
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --mask-dirs <left_mask_dir> <right_mask_dir> \   # optional
    --start-frame 210 --max-frames 160 --interval 4 \
    --model-path weights/vggt_omg/vggt_omg_64x640x480_bf16.pt2 \
    --output-dir output/stream_stereo_vggt_omega \
    --video                                # render trajectory video after the run

# RGB-D (Any2Full): RGB folder + raw-depth folder pair
python tools/general_test/run_stream.py \
    --backend a2f \
    --input-dirs <rgb_frames_dir> \
    --depth-dirs <raw_depth_dir> \
    --start-frame 0 --max-frames 160 --interval 1 \
    --output-dir output/stream_rgbd_a2f
```

| Argument | Default | Description |
|---|---|---|
| `--backend` | **required** | `da3`, `vggt_omega`, or `a2f` (RGB-D) |
| `--input-dirs` | **required** | Can be 1 or several Image folders, one per camera; all must contain the same synchronized frame stems. For `a2f`, the RGB camera folders — the raw depth goes in `--depth-dirs` |
| `--mask-dirs` | — | Optional binary motion-mask folders (one per `--input-dirs`, same order). Pixels above 127 are moving; their confidence is zeroed **during chunk alignment only** — depth outputs are unaffected |
| `--depth-dirs` | — | `a2f` only: raw-depth folders (one per `--input-dirs`, same order and frame stems; `.lz4` uint16 mm maps, 0 = invalid), loaded via `load_depth_lz4` |
| `--depth-scale` | `0.001` | `a2f` only: metres per unit of the raw depth (uint16 mm → metres) |
| `--start-frame` | `0` | First frame to process |
| `--max-frames` | all | Max frames per camera to process |
| `--interval` | `1` | Subsample the sequence (every N-th frame) |
| `--model-path` | backend default | VGGT-Omega artifact (`.pt` / `.pt2`) override |
| `--anyview-model-path` / `--metric-model-path` | backend default | DA3 any-view / metric-depth `.pt2` overrides |
| `--a2f-model-path` | backend default | `a2f` only: Any2Full `.pt2` override |
| `--no-depth-enhance` | off | `a2f` only: skip the Any2Full depth-enhance model and feed the raw sensor depth directly to the alignment |
| `--no-compile` | off | DA3/a2f only: run the exported programs without `torch.compile` (slower, but useful for debugging numeric differences) |
| `--config` | `src/depth_models/streaming/configs/base_config.yaml` | Alignment library / method, loop-closure settings |
| `--output-dir` | `./exps/<backend>_stream_<timestamp>` | Output directory |
| `--device` | auto | `cuda` or `cpu` |
| `--video` / `--video-out` / `--video-fps` / `--video-size` / `--video-max-points` | off | Render a trajectory mp4 after the run (default `<output-dir>/trajectory.mp4`, 10 fps, 960x540, ≤100k points/frame) |

**Output** — per camera: `<output-dir>/depth_<camera_name>/frame_<idx>.lz4`
(depth, raw uint16 mm) + `frame_<idx>.npz` (pose: `extrinsics` 3×4
world→camera, `intrinsics` 3×3, plus `shape` — the depth shape the lz4 buffer
is reshaped to), plus `timings.json`. For `--backend a2f` only the RGB
folders emit `depth_<name>/` — the depth folders are inputs only. The run
finishes with a timing/memory summary: total time, per-chunk mean/min/max,
chunks/s, peak GPU memory (inference only) and peak CPU RSS (process
lifetime).

Wrappers: `scripts/general_test/infer_stream_stereo.sh` (stereo, DA3 /
VGGT-Omega) and `scripts/general_test/infer_stream_rgbd.sh` (RGB-D, a2f).

### visualize_stream.py

> Full documentation: `docs/visualize_stream.md` (viewpoints, fov
> auto-fit, tone-curve inversion, tuning flags).

Renders the streaming output as a trajectory mp4 **without re-running
inference** (reads the saved `.lz4` depth + `.npz` pose outputs). Each frame shows that time step's
coloured point cloud with its camera frustums, plus the growing camera path
from the first frame to the current one. The view is fixed for the whole
video (aligned to the first camera, fitted to the union of all frame clouds);
with `--views 4` each frame is a 2x2 grid of viewpoints (center / down /
left / right), all looking at the scene centre, each viewport's field of
view auto-fitted so the scene fills the frame.

```bash
python tools/general_test/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/stream_stereo_vggt_omega \
    --output output/stream_stereo_vggt_omega/trajectory.mp4 \
    --fps 30 --size 960x540
```

| Argument | Default | Description |
|---|---|---|
| `--input-dirs` | stereo-left/right demo paths | Source image folders, same order as inference |
| `--result-dir` | `output/stream_stereo` | Streaming output directory (the `--output-dir` of `run_stream.py`) |
| `--output` | `<result-dir>/trajectory.mp4` | Video output path |
| `--fps` | `10` | Video frame rate |
| `--size` | `960x540` | Video size `WxH` (even dimensions required) |
| `--max-points` | `100_000` | Max point-cloud points rendered per frame |

Requires `open3d` + `imageio`; the module is also used internally by
`run_stream.py --video` (lazy import, so streaming-only runs don't need
open3d).

Wrapper: `scripts/visualize_stream.sh`.

</details>

<details>
<summary><strong>6. 3D point tracking (TAPIP3D) — `infer_tapip3d.py`</strong></summary>

Streaming long-video 3D point tracking on the TAPIP3D ONNX models — the
encoder/updater/corr_forward run as ONNX sessions in sliding windows of
`seq_len` frames. Image size and query count are auto-detected from the ONNX
graphs. The geometry (`--npz_dir`) is a `run_stream.py`-style output folder:
per frame a `frame_<idx>.lz4` (depth, raw uint16 mm) plus a `frame_<idx>.npz`
pose file (`extrinsic` 4×4 world→camera, `intrinsic` 3×3, `shape` — the depth
shape the lz4 buffer is reshaped to).

```bash
python tools/general_test/infer_tapip3d.py \
    --encoder weights/tapip3d/tapip3d_encoder_480x640.onnx \
    --updater weights/tapip3d/tapip3d_updater.onnx \
    --corr_forward weights/tapip3d/tapip3d_corr_forward.onnx \
    --image_dir <frames_dir> \
    --npz_dir <geometry_dir> \
    --output_dir output/tapip3d_stereo_left_onnx \
    --start_frame 210 --fps 4 \
    --bbox 200 265 260 300 \            # x0 y0 x1 y1 of the tracked object
    --grid_x 8 --grid_y 8 --support_grid_size 32 \
    --num_iters 6 --vis_threshold 0.5 --visualize
```

| Argument | Default | Description |
|---|---|---|
| `--image_dir` | **required** | Folder of frames to track in |
| `--npz_dir` | **required** | Geometry folder: per-frame `frame_<idx>.lz4` (depth, raw uint16 mm) + `frame_<idx>.npz` with `extrinsic` (4×4, world→camera), `intrinsic` (3×3), `shape` |
| `--output_dir`, `-o` | `output/stream_tracks_onnx` | Output directory |
| `--encoder` / `--updater` / `--corr_forward` | `weights/tapip3d/*.onnx` | ONNX model paths |
| `--start_frame` / `--max_frames` / `--fps` | `0` / all / `1` | Sequence subsampling (e.g. `--fps 4` tracks at 4 Hz on 30 Hz data) |
| `--bbox` | full frame | `x0 y0 x1 y1` of the tracked object; a grid of points inside is unprojected to world-space queries with the first frame's depth + pose |
| `--grid_x` / `--grid_y` | `8` / `8` | Bbox grid points |
| `--support_grid_size` | `32` | Full-frame support grid (added to the bbox queries) |
| `--num_iters` | `6` | Update iterations inside each window |
| `--vis_threshold` | `0.5` | Sigmoid visibility threshold for `visibs` |
| `--visualize` / `--video_fps` | off / `10` | Render the tracks as a video after inference |

> **Exact-N contract**: the shipped updater ONNX has a **fixed query count
> (1088)**. The default grids produce exactly that: 8×8 bbox grid (64) +
> 32×32 support grid (1024). The script exits with a clear error otherwise
> (e.g. depth holes in the bbox drop bbox points).

**Output** — `coords.npy` (T, Q, 3) world-space 3D traces, `visibs.npy`
(T, Q) per-point visibility, `metadata.json` (paths, grid, bbox, frame
indices), and with `--visualize` a video of the tracks.

Wrapper: `scripts/infer_tapip3d.sh`.

</details>

<details>
<summary><strong>Relation to dataset-specific tools</strong></summary>

The scripts in this folder are the **general-purpose reference entry
points**. Dataset-specific tool folders reuse their functions but optimize
for a particular dataset:

- `tools/astribot` — Astribot dataset helpers (e.g. frame extraction).
- `tools/hifi-umi` — HiFi-UMI dataset preprocessing (`extract_frames.py`,
  `generate_masks.py`), reusing the same model entry points.

See `docs/waft.md` for WAFT-specific model notes; the repo README documents
the end-to-end pipeline built on these tools.

</details>
