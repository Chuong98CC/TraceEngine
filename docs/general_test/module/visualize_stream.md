# Trajectory Video (`tools/general_test/pipeline/visualize_stream.py`)

Renders the streaming depth+pose output as a **trajectory mp4** without
re-running inference — it only reads the saved per-frame geometry
(`.lz4` depth + `.npz` pose, the contract written by `run_depth_stream.py`).
Each video frame shows that time step's coloured point cloud with its
camera frustums, plus the growing camera path from the first frame to the
current one. The view is **fixed for the whole video** (aligned to the
first camera, fitted to the union of all frame clouds), so the scene
stays put while per-frame content moves inside it.

- One viewpoint by default (behind the first camera), or a **2×2 grid**
  of four viewpoints (`--views 4`, default): center / down / left /
  right — all looking at the scene centre.
- Each viewport's field of view is **auto-fitted** so the scene fills the
  frame (~85% of the viewport, per-viewport).
- Used in three places: the standalone CLI below,
  `run_depth_stream.py --video` (rendered after the run, lazy-imported so
  streaming-only runs don't need open3d), and the online per-sub-task
  variant `tools/astribot/visualize_subtask_stream.py` (see
  `astribot_visualize_subtask_depth_stream.md`), which decodes the colour
  frames from the LeRobotDataset instead of reading frame folders and
  exposes the same `--view-*` tuning flags.

## How it works

**Inputs.** Per camera, `<result-dir>/depth_<camera_name>/` contains
`<stem>.lz4` (depth, raw uint16 mm) + `<stem>.npz` (`extrinsics` 3×4
world→camera, `intrinsics` 3×3, `shape` — the depth shape the lz4 buffer
is reshaped to), loaded via `utils.streaming_utils.load_stream_data`.
The colour images (`<stem>.jpg/.jpeg/.png`) come from the `--input-dirs`
folders and are resized to the depth resolution. Time order = the sorted
NPZ stems of the first camera's output folder.

**View construction.** A fixed `4×4` alignment transform puts the first
camera at the origin in glTF axes (looking along `-z`, up `+y`) and
centres the scene on the median of the union of all frame clouds +
trajectory. All eyes are expressed in this frame and look at the scene
centre (the aligned origin):

- **center** — `view_distance` scene-extents behind the first camera
  (`--view-distance`, default `0.3`).
- **down** — the center eye raised `view_raise` extents and pulled back
  `view_back` extents (see *Pose in frame* below).
- **left / right** — the center eye's vector swung ±`view_angle` degrees
  around the scene centre's vertical axis (default `45`, clamped ≤85),
  lowered `view_lower` extents — a near side-on look that keeps the
  center eye's distance from the scene.

**Framing.** Per viewport, the union scene points are projected into the
eye's camera frame; the 90th-percentile angular half-extent per axis is
mapped to ~85% of the viewport half-angle, clamped to 20–90° vertical.
`--view-fov` overrides the fit (vertical degrees, applied to all
viewports).

**Pose in frame.** The auto-fit frames the **cloud**; the camera pose
(frustums + path) often falls outside the fitted viewport for the
off-center views — the first camera's apex is ~29° off the down eye's
gaze and ~46° off a 45°-swung side eye's gaze, vs a fitted half-fov of
~20°. That's why the down eye is pulled back (`--view-back`, 0.3 extents
brings the apex inside the fitted frame) and the side eyes are lowered
(`--view-lower`). If the pose is cut in your data, raise these flags
(side views need ~0.6 extents of pull-back before the frustums fully
enter, at the cost of swing).

**Colours.** The open3d renderer applies a filmic tone curve that cannot
be disabled, so `_build_inverse_tone_curve` inverts it empirically: a
dense gray point grid is rendered at each gray level and the center pixel
is read back into a LUT (cached per open3d version). Video colours match
the source images.

**Encoding.** Each frame is rendered offscreen (`OffscreenRenderer`) and
fed to imageio-ffmpeg (`libx264`, `macro_block_size=1`); the video size
is snapped to even dimensions (x264 requirement).

## Usage

### 1. Standalone re-render of saved outputs

```bash
python tools/general_test/pipeline/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/stream_stereo_vggt_omega \
    --output output/stream_stereo_vggt_omega/trajectory.mp4 \
    --fps 30 --size 960x540
```

Wrapper: `scripts/general_test/visualize_stream.sh`.

### 2. Single view instead of the 2×2 grid

```bash
python tools/general_test/pipeline/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/stream_stereo_vggt_omega \
    --views 1
```

### 3. Tuning the viewpoints

| Want… | Flag |
|---|---|
| side views more side-on / closer to center | `--view-angle` (up to 85; smaller = closer to center) |
| side views lower (more table-level) | `--view-lower` |
| down view more top-down | `--view-raise` |
| down view pulled further back (pose into frame) | `--view-back` |
| zoom out / in (all viewports) | `--view-distance` (eye distance in scene extents) |
| manual framing instead of auto-fit | `--view-fov` (vertical degrees) |

## Output

`<output>` (default `<result-dir>/trajectory.mp4`), H.264/yuv420p at the
exact requested size (even dimensions), `fps` frames per second.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--input-dirs` | demo stereo paths | Source image folders (one per camera), same order as inference |
| `--result-dir` | `output/stream_stereo` | Streaming output directory (the `--output-dir` of `run_depth_stream.py`) |
| `--output` | `<result-dir>/trajectory.mp4` | Video output path |
| `--fps` | `10` | Video frame rate |
| `--size` | `960x540` | Video size `WxH` (snapped to even) |
| `--max-points` | `100_000` | Max point-cloud points rendered per video frame |
| `--view-distance` | `0.3` | Eye distance behind the first camera, in scene-extent units |
| `--views` | `4` | `1` (center only) or `4` (2×2 grid: center/down/left/right) |
| `--view-angle` | `45.0` | Side-view swing for left/right, degrees off the center view (clamped ≤ 85) |
| `--view-lower` | `0.1` | Downward shift of the left/right viewpoints, in scene-extent units |
| `--view-raise` | `0.1` | Elevation of the down viewpoint, in scene-extent units |
| `--view-back` | `0.3` | Backward pull of the down viewpoint, in scene-extent units (keeps the camera pose inside the fitted frame) |
| `--view-fov` | auto | Override the auto-fitted vertical field of view, in degrees |

`render_stream_video(...)` (the underlying function) additionally takes
`stride` (subsample the sequence for the view fitting only) and
`frame_loader` (online images instead of frame folders) — used by
`visualize_subtask_stream.py`, not exposed on this CLI.

## Notes

- Requires `open3d` + `imageio` (offscreen renderer). `run_depth_stream.py`
  imports the module lazily behind `--video`.
- The tone-LUT build renders a few hundred small frames once per process
  (a second or two of startup overhead).
- Outputs written by older pipeline versions (depth stored inside the
  NPZ, no `.lz4` / `shape`) are not readable by the current loaders.
- Related docs: [`streaming.md`](streaming.md) (the `run_depth_stream.py`
  inference side), [`astribot_visualize_subtask_depth_stream.md`](../../astribot/astribot_visualize_subtask_depth_stream.md)
  (online per-sub-task variant).
