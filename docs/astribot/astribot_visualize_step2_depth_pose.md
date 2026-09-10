# Per-Sub-Task Depth+Pose Videos (`tools/astribot/visualize_step2_depth_pose.py`)

Renders **one `depth_pose.mp4` per (sub-task, selected camera)** of an
episode, online — the visualization counterpart of
`tools/astribot/run_step2_depth_stream.py` (see `astribot_subtask_stream.md`).
It does **not** re-run inference: the geometry (depth from the saved `.lz4`
files, pose from the paired `.npz` files) is read from the saved depth_pose
outputs, and the colour images shown in the point clouds are **decoded
online from the LeRobotDataset** (one frame per rendered step), so no
extracted frames or videos need to exist on disk.

- Standalone (no `DataExtract`): the episodes, sub-task segments, cameras
  and rendered steps are all discovered from the saved Step-2 outputs on
  disk (`<out-dir>/depth_pose/ep*/subtask_XX/depth_<camera>/`) — the
  on-disk segmentation, so there is no dataset split inference and no
  selection flag that must match the `run_step2_depth_stream.py` run;
  `-e`/`-c` merely filter what exists.
- Built on `tools/general_test/pipeline/visualize_stream.py`'s `render_stream_video`
  (the same lz4-depth + npz-pose input contract), with a `frame_loader` that
  decodes the segment's dataset frames on the fly instead of reading frame
  folders. The view construction, framing and `--view-*` tuning flags are
  shared with it — see [`visualize_stream.md`](../general_test/module/visualize_stream.md)
  for the details.
- Output: one `depth_pose.mp4` per selected camera that has Step-2 outputs —
  `<out-dir>/visualization/<episode>/subtask_XX/<camera>/depth_pose.mp4` —
  each video frame showing that step's coloured point cloud with the
  camera's frustum and the growing camera path; the view is fixed per
  segment (aligned to the first camera, fitted to the union of the
  segment's clouds). A multi-camera Step-2 run therefore yields one video
  per camera, each in its own folder.

## How it works

**Inputs.** Geometry comes from `run_step2_depth_stream.py`'s outputs:
`<out-dir>/depth_pose/<episode>/subtask_XX/depth_<camera>/frame_<idx>.lz4`
(absolute dataset indices, stride-subsampled). Each `.lz4` holds the depth
as log-encoded uint8 — float metres over [0.001, 2.001] m, decoded by
`load_depth_lz4` — reshaped with the `shape` recorded in the paired
`frame_<idx>.npz` (`extrinsics` 3×4 world→camera, `intrinsics` 3×3) — the
same contract as `utils.streaming_utils.load_stream_data`. A selected camera
without saved results for a segment is skipped with a message.

**Online frames.** For each rendered step of a camera's video, the dataset
frame at the stem's absolute index is decoded for that camera (BGR→RGB
flip, resized to the camera's depth resolution — matching `load_pair`'s
image contract). Only the rendered steps' frames are decoded, so memory
stays flat.

**Rendering.** `render_stream_video` does the offscreen rendering: per step
the coloured point cloud (subsampled to `--max-points`), that step's camera
frustum, and the trajectory line of all steps up to the current one. By
default each frame is a **2×2 grid** of four viewpoints — center / down /
left / right — all fixed for the segment (aligned to the first camera) and
each auto-fitted so the scene fills the frame. The `--view-*` flags tune the
viewpoints: `--view-angle` / `--view-lower` for the side views, `--view-raise`
/ `--view-back` for the down view (pull it back if the camera pose is cut
off), `--view-fov` to override the auto-fit, `--views 1` for a single center
view. Output is encoded straight to `depth_pose.mp4` (H.264, even
dimensions).

## Usage

### 1. All sub-tasks of episode 0, every camera with Step-2 outputs

```bash
python tools/astribot/visualize_step2_depth_pose.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Default cameras: every camera with Step-2 outputs on disk (`cam_head`,
`cam_head_stereo_left`, `cam_head_stereo_right` and `cam_torso` on this
dataset) — one `depth_pose.mp4` per camera.

### 2. Stereo pair at 30 fps (one video per camera)

```bash
python tools/astribot/visualize_step2_depth_pose.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0 \
    --camera-idxes 4 5 --fps 30
```

`-c` merely filters what exists: results are read from the same
`depth_<camera>` folders, so a camera the Step-2 run skipped for a segment
is passed over with a message.

## Output

```
<out-dir>/visualization/              # <out-dir> defaults to <data-root>/eps_data
└── ep000000/
    └── subtask_00/
        ├── cam_head/                 # one dir per selected camera with
        │   └── depth_pose.mp4        #   Step-2 outputs ← produced here
        └── cam_head_stereo_left/
            └── depth_pose.mp4
        └── …
```

The Step-4 trace videos (`trace2d.mp4` / `trace3d.mp4`,
`visualize_step4_traces.py` — see `astribot_traces.md`) land in the same
per-camera folders, so each camera's depth+pose stream and its traces sit
together.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root passed to LeRobotDataset; full chunk path for chunked datasets |
| `--camera-idxes`, `-c` | every camera with Step-2 outputs | indices into the dataset's `camera_keys` to visualize; only cameras with saved `depth_<camera>` results are rendered |
| `--episode-idxes`, `-e` | all on-disk | only visualize these episode indices (every episode with Step-2 outputs under `<out-dir>/depth_pose` by default); a requested episode without outputs is warned about and skipped |
| `--max-episodes`, `-x` | — | cap the number of discovered episodes (first N, after any `-e` filter) |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root; results read from `<out-dir>/depth_pose/<episode>/subtask_XX/`, videos written under `<out-dir>/visualization/` |
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
