# Astribot Episode Splitting (`tools/astribot/extract_frames.py`)

Extracts **sub-task splits**, **key-frame jpgs** and/or **per-subtask videos**
from a local [LeRobotDataset](https://github.com/huggingface/lerobot) copy of
an Astribot dataset. It is the dataset-preprocessing step that turns long
multi-subtask episodes into clean per-subtask segments for downstream
training / evaluation.

```
detect_subtask  →  subtask_splits.json + gripper plot   (tabular only, no video)
key_frames      →  one jpg per first/last/key frame, per sub-task (decodes videos)
videos          →  one mp4 per sub-task segment         (ffmpeg cut, no decode)
frames          →  sampled per-subtask jpgs + uint16 depth lz4 (decodes videos)
```

The four modes are independent: `detect_subtask` only reads the tabular data
(so it also runs on datasets whose videos are not on disk), while
`key_frames`, `videos` and `frames` need the dataset videos on disk
(`LeRobotDataset` is opened with `download_videos=False`).

## How the split frames are decided

- **Ground truth first.** The `videos`, `frames` and `key_frames` modes
  prefer the dataset's per-frame `subtask_index` column when it exists: the
  split frames are simply collected at **every frame where the subtask
  changes** (the first frame of each new sub-task segment).
- **Fallback.** Only when the episode has **fewer than two sub-tasks** (no
  change anywhere, or no `subtask_index` column) are the split frames loaded
  from the `subtask_splits.json` written by the `detect_subtask` mode.
- **Gripper inference.** `detect_subtask` itself infers the key frames and
  the splits from the `observation.state` gripper columns only: key frames
  are the first/last frame plus every frame where the smoothed gripper
  signal crosses the open/close threshold, and a split is placed at the
  midpoint of each complete `closed → open → re-closed` (1 → 0 → 1) pattern.
  The plot (`split_graph_<ep>.png`) shows the raw + filtered gripper state,
  the detected key frames and the inferred split frames, with the
  ground-truth subtask index as a second subplot when available. This mode
  needs gripper columns in `observation.state`; without them it warns and
  saves an empty split list.
- **Key frames.** The `key_frames` mode saves the episode's first and last
  frame (the ground-truth subtask bounds when the dataset has
  `subtask_index`) plus the gripper-detected key frames — ground truth has
  no key frames, so they always come from `detect_subtask`. Each jpg is
  saved under the sub-task segment its frame belongs to (an episode without
  splits lands everything in `subtask_00`).

## Requirements

- A **local copy** of the dataset (LeRobotDataset layout: `meta/`, per-episode
  parquet, and `videos/` for the video-consuming modes). `--data-root` is
  the root passed to `LeRobotDataset`; for chunked datasets give the **full
  chunk path**, e.g. `/data/x/chunk-0000/part-0000`.
- `uv sync` done in the repo root (the script imports from `utils`).
- `ffmpeg` on `PATH` for the `video` mode.
- No model weights are involved — this tool only touches the dataset.

## Usage

### 1. If the dataset does not have subtask_index, infer the sub-task splits first.

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 \
    --mode detect_subtask
```

Writes per episode:

```
<out_dir>/subtask/ep000000/
├── subtask_splits.json        # episode, task_id, task, from_idx, to_idx,
│                              # key_frames[], split_frames[]
└── split_graph_000000.png     # gripper state + key/split frames plot
```

### 2. Save the first/last/key frames as jpgs

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 \
    --mode key_frames \
    --camera-idxes 0 1          # indices into the dataset's camera_keys
```

Output: `<out_dir>/key_frames/ep000000/subtask_XX/<camera>/frame_<idx>.jpg`
of the episode's first, last and key frames (key frames from
`detect_subtask`), grouped by the sub-task segment each frame belongs to.

### 3. Cut one mp4 per sub-task segment

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 \
    --mode videos \
    --camera-idxes 0 1
```

Output: `<out_dir>/subtask_videos/ep000000/<camera>/subtask_00.mp4`,
`subtask_01.mp4`, … — the episode cut at the split frames. The cuts are
**always frame-accurate**: each segment is re-encoded with x264 at its
exact split frames (a decode+encode pass is unavoidable for exact H.264
cuts — stream copy can only snap to keyframes). Depth cameras are written
as monochrome streams.

### 4. Sample frames of every sub-task

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 \
    --mode frames \
    --camera-idxes 0 --interval 4 --max-frames 50
```

Output per sub-task segment:
`<out_dir>/subtask_frames/ep000000/subtask_00/<camera>/frame_<idx>.jpg` for
every `--interval`-th frame, capped at `--max-frames` per sub-task. When a
camera has a paired depth feature stored as **uint16 (mm)** (here
`observation.depth.cam_head` for `observation.images.cam_head`), the raw
depth array is saved as
`<out_dir>/subtask_frames/ep000000/subtask_00/depth_<camera>/frame_<idx>.lz4`
(raw uint16 mm, loadable by `utils.astribot_dataloader.load_depth_lz4`).
Depth pairing prefers the raw `observation.depth.*` feature; the legacy
`<cam_key>_depth` video is only trusted when the dataset metadata flags it as
a real depth map (`video.is_depth_map`) — the `astri_making_coffee` recording
is flagged false and its depth is unusable (see the re-recorded
`astri_making_coffee_v1`).

### Typical workflow

```bash
# 1) splits + key frames (no video needed) — then optionally inspect the plots
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m detect_subtask

# 2) jpgs, videos and/or sampled frames, from the ground-truth
#    subtask_index when present, otherwise from subtask_splits.json
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m key_frames -c 0 1
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m videos -c 0 1
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m frames -c 0 1
```

Note that when the dataset **has** a `subtask_index` column, step 2 runs
without the json produced by step 1 (the splits come from the ground truth
directly).

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root passed to LeRobotDataset; full chunk path for chunked datasets |
| `--mode`, `-m` | — (required) | `detect_subtask` / `key_frames` / `videos` / `frames` |
| `--camera-idxes`, `-c` | `[2, 1]` | indices into the dataset's `camera_keys` |
| `--episode-idxes`, `-e` | all episodes | only these episode indices (mutually exclusive with `--one-per-task`) |
| `--one-per-task` | off | process only the first episode of each task |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--interval` | `4` | `frames` mode: sample every N-th frame of each sub-task segment |
| `--max-frames` | all | `frames` mode: cap the sampled frames per sub-task |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root holding `subtask/`, `key_frames/`, `subtask_videos/`, `subtask_frames/` |
| `--dedup-tasks` | off | skip episodes whose task already produced output (useful across runs) |
| `--min-close-seconds` | `1.5` | closed-gripper runs shorter than this are sensor noise (key-frame detection) |

## Notes

- On startup the tool prints a meta summary: episode/frame counts, the camera
  table (with the selected cameras marked), the `observation.state` layout and
  the subtask index → description mapping from `meta/subtasks.parquet` when
  the dataset has one.
- If `key_frames` / `videos` / `frames` raise "subtask_splits.json missing:
  run --mode detect_subtask first", the episode has no usable ground-truth
  `subtask_index` (or fewer than two sub-tasks) — run the `detect_subtask`
  mode once so the fallback json exists. `key_frames` always needs the json
  (its key frames come from the gripper analysis only).
- The `frames` mode only saves depth when the paired feature is stored as
  uint16 (mm); when the only depth source is a video flagged
  `video.is_depth_map=false`, a warning is printed and depth is skipped.
- A wrapper for a single video cut is in `scripts/extract_frames.sh`.
