# Astribot Episode Splitting (`tools/astribot/extract_frames.py`)

Extracts **sub-task splits**, **key-frame jpgs** and/or **per-subtask videos**
from a local [LeRobotDataset](https://github.com/huggingface/lerobot) copy of
an Astribot dataset. It is the dataset-preprocessing step that turns long
multi-subtask episodes into clean per-subtask segments for downstream
training / evaluation.

```
split_subtask   →  subtask_splits.json + gripper plot   (tabular only, no video)
extract_frames  →  one jpg per split frame per camera   (decodes videos)
video           →  one mp4 per sub-task segment         (ffmpeg cut, no decode)
```

The three modes are independent: `split_subtask` only reads the tabular data
(so it also runs on datasets whose videos are not on disk), while
`extract_frames` and `video` need the dataset videos on disk
(`LeRobotDataset` is opened with `download_videos=False`).

## How the split frames are decided

- **Ground truth first.** The `extract_frames` and `video` modes prefer the
  dataset's per-frame `subtask_index` column when it exists: the split frames
  are simply collected at **every frame where the subtask changes** (the first
  frame of each new sub-task segment).
- **Fallback.** Only when the episode has **fewer than two sub-tasks** (no
  change anywhere, or no `subtask_index` column) are the split frames loaded
  from the `subtask_splits.json` written by the `split_subtask` mode.
- **Gripper inference.** `split_subtask` itself infers the splits from the
  `observation.state` gripper columns only: key frames are the first/last
  frame plus every frame where the smoothed gripper signal crosses the
  open/close threshold, and a split is placed at the midpoint of each
  complete `closed → open → re-closed` (1 → 0 → 1) pattern. The plot
  (`split_graph_<ep>.png`) shows the raw + filtered gripper state, the
  detected key frames and the inferred split frames, with the ground-truth
  subtask index as a second subplot when available. This mode needs gripper
  columns in `observation.state`; without them it warns and saves an empty
  split list.

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
    --data-root /data/astri_making_coffee \
    --mode split_subtask
```

Writes per episode:

```
<out_dir>/subtask/ep000000/
├── subtask_splits.json        # episode, task_id, task, from_idx, to_idx,
│                              # key_frames[], split_frames[]
└── split_graph_000000.png     # gripper state + key/split frames plot
```

### 2. Save the split frames as jpgs

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --mode extract_frames \
    --camera-idxes 0 1          # indices into the dataset's camera_keys
```

Output: `<out_dir>/key_frames/ep000000/<camera>/frame_<idx>.jpg` per split
frame and selected camera (depth cameras are saved grayscale).

### 3. Cut one mp4 per sub-task segment

```bash
python tools/astribot/extract_frames.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --mode video \
    --camera-idxes 0 1
```

Output: `<out_dir>/subtask_videos/ep000000/<camera>/subtask_00.mp4`,
`subtask_01.mp4`, … — the episode cut at the split frames. By default ffmpeg
**stream-copies** (`-c copy`, fast, boundaries snap to the nearest keyframes);
pass `--exact` to re-encode with x264 for frame-accurate boundaries (depth
cameras are written as monochrome streams).

### Typical workflow

```bash
# 1) splits (no video needed) — then optionally inspect the plots
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m split_subtask

# 2) jpgs and/or videos, from the ground-truth subtask_index when present,
#    otherwise from subtask_splits.json
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m extract_frames -c 0 1
python tools/astribot/extract_frames.py -id <repo> -d <data_root> -m video -c 0 1
```

Note that when the dataset **has** a `subtask_index` column, step 2 runs
without the json produced by step 1 (the splits come from the ground truth
directly).

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root passed to LeRobotDataset; full chunk path for chunked datasets |
| `--mode`, `-m` | — (required) | `split_subtask` / `extract_frames` / `video` |
| `--camera-idxes`, `-c` | `[2, 1]` | indices into the dataset's `camera_keys` |
| `--episode-idxes`, `-e` | all episodes | only these episode indices (mutually exclusive with `--one-per-task`) |
| `--one-per-task` | off | process only the first episode of each task |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root holding `subtask/`, `key_frames/`, `subtask_videos/` |
| `--dedup-tasks` | off | skip episodes whose task already produced output (useful across runs) |
| `--exact` | off | x264 re-encode instead of `-c copy` in the video mode |
| `--min-close-seconds` | `1.5` | closed-gripper runs shorter than this are sensor noise (key-frame detection) |

## Notes

- On startup the tool prints a meta summary: episode/frame counts, the camera
  table (with the selected cameras marked), the `observation.state` layout and
  the subtask index → description mapping from `meta/subtasks.parquet` when
  the dataset has one.
- If `extract_frames` / `video` raise "subtask_splits.json missing: run
  --mode split_subtask first", the episode has no usable ground-truth
  `subtask_index` (or fewer than two sub-tasks) — run the `split_subtask`
  mode once so the fallback json exists.
- A wrapper for a single video cut is in `scripts/extract_frames.sh`.
