# Streaming Depth + Pose (`tools/general_test/pipeline/run_depth_stream.py` & `visualize_stream.py`)

Multi-camera depth + camera pose over a long trajectory, in sliding chunks
aligned into one world frame.

## run_depth_stream.py

Unified entry point for three backends:

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
  and the any-view depth is aligned to it (see [`any2full.md`](any2full.md)).
  Only the RGB folders emit output (`depth_<name>/`) — the depth folders are
  inputs only. With `--no-depth-enhance` Any2Full is not loaded and the raw
  sensor depth feeds the alignment directly.

`chunk_size` is derived from the model's fixed `num_views` (read from the
export itself) and must be divisible by the number of input folders. A short
final chunk is padded at its start with images from the previous chunk; the
padded outputs are discarded.

```bash
# Stereo (VGGT-Omega or DA3)
python tools/general_test/pipeline/run_depth_stream.py \
    --backend vggt_omega \                 # or "da3"
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --mask-dirs <left_mask_dir> <right_mask_dir> \   # optional
    --start-frame 210 --max-frames 160 --interval 4 \
    --model-path weights/vggt_omg/vggt_omg_64x640x480_bf16.pt2 \
    --output-dir output/stream_stereo_vggt_omega \
    --video                                # render trajectory video after the run

# RGB-D (Any2Full): RGB folder + raw-depth folder pair
python tools/general_test/pipeline/run_depth_stream.py \
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
| `--mask-dirs` | — | Optional binary motion-mask folders (one per `--input-dirs`, same order). Pixels above 127 are moving; their confidence is zeroed **during chunk alignment only** — depth outputs are unaffected (masks from `infer_waft.py`, see [`waft.md`](waft.md)) |
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

This output contract is what `infer_tapip3d.py` consumes for 3D tracing
(see [`tapip3d.md`](tapip3d.md)) and what `visualize_rgbd.py` /
`visualize_stream.py` render.

## visualize_stream.py

> Full documentation: [`visualize_depth_stream.md`](visualize_depth_stream.md)
> (viewpoints, fov auto-fit, tone-curve inversion, tuning flags).

Renders the streaming output as a trajectory mp4 **without re-running
inference** (reads the saved `.lz4` depth + `.npz` pose outputs). Each frame shows that time step's
coloured point cloud with its camera frustums, plus the growing camera path
from the first frame to the current one. The view is fixed for the whole
video (aligned to the first camera, fitted to the union of all frame clouds);
with `--views 4` each frame is a 2x2 grid of viewpoints (center / down /
left / right), all looking at the scene centre, each viewport's field of
view auto-fitted so the scene fills the frame.

```bash
python tools/general_test/pipeline/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/stream_stereo_vggt_omega \
    --output output/stream_stereo_vggt_omega/trajectory.mp4 \
    --fps 30 --size 960x540
```

| Argument | Default | Description |
|---|---|---|
| `--input-dirs` | stereo-left/right demo paths | Source image folders, same order as inference |
| `--result-dir` | `output/stream_stereo` | Streaming output directory (the `--output-dir` of `run_depth_stream.py`) |
| `--output` | `<result-dir>/trajectory.mp4` | Video output path |
| `--fps` | `10` | Video frame rate |
| `--size` | `960x540` | Video size `WxH` (even dimensions required) |
| `--max-points` | `100_000` | Max point-cloud points rendered per frame |

Requires `open3d` + `imageio`; the module is also used internally by
`run_depth_stream.py --video` (lazy import, so streaming-only runs don't need
open3d).

## Wrapper scripts (`scripts/general_test/`)

- `infer_stream_stereo.sh` — stereo streaming (DA3 / VGGT-Omega) on the
  Astribot sample extraction, then renders the trajectory video.
- `infer_stream_rgbd.sh` — RGB-D streaming (`a2f` backend) on the Astribot
  sample extraction, then renders the trajectory video.
- `visualize_stream.sh` — trajectory video from a streaming result.
