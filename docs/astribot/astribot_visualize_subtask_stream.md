# Per-Sub-Task Trajectory Video (`tools/astribot/visualize_subtask_stream.py`)

Renders **one trajectory video per sub-task segment** of an episode, online —
the visualization counterpart of `tools/astribot/run_step2_depth_stream.py` (see
`astribot_subtask_depth_stream.md`). It does **not** re-run inference: the geometry
(depth from the saved `.lz4` files, pose from the paired `.npz` files) is read
from the pipeline outputs, and the colour images shown in the point clouds are
**decoded online from the LeRobotDataset** (one frame per rendered step), so
no extracted frames or videos need to exist on disk.

- Reuses `DataExtract` (`tools/astribot/extract_frames.py`, see
  `astribot_extract_frames.md`) for dataset introspection, episode/camera
  selection and sub-task split-frame inference — the selection flags are
  identical to `run_step2_depth_stream.py`, so the same episode/camera command
  line works for both tools.
- Built on `tools/general_test/pipeline/visualize_stream.py`'s `render_stream_video`
  (the same lz4-depth + npz-pose input contract), with a `frame_loader` that
  decodes the segment's dataset frames on the fly instead of reading frame
  folders. The view construction, framing and `--view-*` tuning flags are
  shared with it — see `visualize_depth_stream.md` for the details.
- Output: `<seg_dir>/trajectory.mp4` per segment — each video frame shows that
  step's coloured point cloud with its camera frustums and the growing camera
  path; the view is fixed per segment (aligned to the first camera, fitted to
  the union of the segment's clouds).

## How it works

**Inputs.** Geometry comes from `run_step2_depth_stream.py`'s outputs:
`<out-dir>/pipeline/<episode>/subtask_XX/depth_<camera>/frame_<idx>.lz4`
(absolute dataset indices, stride-subsampled). Each `.lz4` holds the depth as
raw uint16 mm and is reshaped with the `shape` recorded in the paired
`frame_<idx>.npz` (`extrinsics` 3×4 world→camera, `intrinsics` 3×3) — the
same contract as `utils.streaming_utils.load_stream_data`. Segments without
saved results are skipped with a message.

**Online frames.** For each rendered step, the dataset frame at the stem's
absolute index is decoded for every selected camera (BGR→RGB flip, resized to
the depth resolution — matching `load_pair`'s image contract). Only the
rendered steps' frames are decoded, so memory stays flat.

**Rendering.** `render_stream_video` does the offscreen rendering: per step
the coloured point cloud (subsampled to `--max-points`), that step's camera
frustums, and the trajectory line of all steps up to the current one. By
default each frame is a **2×2 grid** of four viewpoints — center / down /
left / right — all fixed for the segment (aligned to the first camera) and
each auto-fitted so the scene fills the frame. The `--view-*` flags tune the
viewpoints: `--view-angle` / `--view-lower` for the side views, `--view-raise`
/ `--view-back` for the down view (pull it back if the camera pose is cut
off), `--view-fov` to override the auto-fit, `--views 1` for a single center
view. Output is encoded straight to `trajectory.mp4` (H.264, even
dimensions).

## Usage

### 1. All sub-tasks of episode 0, single camera

```bash
python tools/astribot/visualize_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Default camera is the first non-depth camera key (`cam_head` on this
dataset).

### 2. Stereo pair at 30 fps

```bash
python tools/astribot/visualize_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0 \
    --camera-idxes 4 5 --fps 30
```

The cameras must match the ones used by `run_step2_depth_stream.py` — results are
read from the same `depth_<camera>` folders.

## Output

```
<out-dir>/pipeline/                     # <out-dir> defaults to <data-root>/eps_data
└── ep000000/
    ├── subtask_00/
    │   ├── depth_cam_head/             # run_step2_depth_stream.py outputs (read-only)
    │   └── trajectory.mp4              # ← produced here, per segment
    ├── subtask_01/
    │   └── trajectory.mp4
    └── …
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root passed to LeRobotDataset; full chunk path for chunked datasets |
| `--camera-idxes`, `-c` | first non-depth camera | indices into the dataset's `camera_keys`; must match the cameras used by `run_subtask_stream` |
| `--episode-idxes`, `-e` | all | only process these episode indices |
| `--one-per-task` | off | select the first episode of each task (exclusive with `-e`) |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root; results read from `<out-dir>/pipeline/<episode>/subtask_XX/` |
| `--stride` | 4 | subsample the sequence for the view fitting only; the rendered steps are the file stems actually saved |
| `--fps` | 10 | video frame rate |
| `--size` | `960x540` | video size `WxH` (even dimensions required) |
| `--max-points` | `100_000` | max point-cloud points rendered per video frame |
| `--view-distance` | `0.3` | eye distance behind the first camera, in scene-extent units |
| `--views` | `4` | `1` (center only) or `4` (2×2 grid: center/down/left/right) |
| `--view-angle` | `45.0` | side-view swing for left/right, degrees off the center view (clamped ≤ 85) |
| `--view-lower` | `0.1` | downward shift of the left/right viewpoints, in scene-extent units |
| `--view-raise` | `0.1` | elevation of the down viewpoint, in scene-extent units |
| `--view-back` | `0.3` | backward pull of the down viewpoint, in scene-extent units (keeps the camera pose inside the fitted frame) |
| `--view-fov` | auto | override the auto-fitted vertical field of view, in degrees |

## Notes

- Requires `open3d` + `imageio` (the offscreen renderer), plus the
  lz4/npz outputs of `run_step2_depth_stream.py` — the tool only renders, it
  never runs inference.
- The final `AttributeError: '_thread.RLock' …` printed at interpreter
  shutdown is harmless `multiprocess.resource_tracker` noise (exit code 0).
