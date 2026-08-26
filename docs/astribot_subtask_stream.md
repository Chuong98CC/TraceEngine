# Per-Sub-Task Depth + Pose Streaming (`tools/astribot/run_subtask_stream.py`)

Streams **metric depth + camera pose per sub-task segment** directly from a
local [LeRobotDataset](https://github.com/huggingface/lerobot) copy of an
Astribot dataset, with **optional per-chunk WAFT motion masks**. It chains the
pipeline steps described in `general_test.md` and the README — WAFT motion
masks then streaming depth + pose — without ever writing frames to disk:

- Wraps `DataExtract` (`tools/astribot/extract_frames.py`, see
  `astribot_extract_frames.md`) for dataset introspection, episode/camera
  selection and sub-task split-frame inference.
- Frames are **decoded online** from the dataset (`LeRobotDataset` with
  `download_videos=False`), one chunk at a time — no `extract_frames` jpgs,
  no per-subtask mp4s, nothing stored on disk.
- The streaming backend (`VGGT_OMG_Streaming` / `DA3_Streaming`, the same
  models as `run_stream.py`) runs per segment; each segment's output lands in
  its own `subtask_XX/` folder.

## How it works

**Sub-task segments.** For each episode, the split frames come from
`DataExtract._load_splits()` — the ground-truth `subtask_index` column when
the dataset has it, the gripper-inferred `subtask_splits.json` otherwise. The
segment bounds are `[from_idx] + split_frames + [to_idx]`; every segment is
streamed independently (alignment restarts per sub-task).

**Online frames.** `OnlineStreaming` (mixed into the concrete backend) keeps
`img_list` as virtual `frame_%06d.jpg` stems — output npz files are named
after them, so saved frames keep their **absolute dataset indices** (e.g.
`frame_000004.npz`). Before each chunk's forward, the chunk's image arrays are
decoded from the dataset and slice-swapped into `img_list`; they are restored
right after. Peak memory is one chunk of frames + masks.

**Chunk shape.** `chunk_size` is the model's fixed `num_views` (64 for the
shipped VGGT-Omega export, 64 or 24 for the DA3 any-view exports) and must be
**divisible by the number of selected cameras**. One chunk is `num_cams ×
frames_per_chunk` images with `frames_per_chunk = chunk_size // num_cams`, so:

| cameras | chunk (views) | steps/chunk | each camera |
|---|---|---|---|
| 1 | 64 | 64 | 64 views |
| 2 (stereo) | 64 | 32 | **32 views = 1/2 of the chunk** |

Selecting the stereo pair (`--camera-idxes 4 5`) is the equivalent of feeding
`LEFT_DIR RIGHT_DIR` to `run_stream.py --input-dirs` (see
`scripts/infer_stream.sh`): both cameras run jointly through the same forward,
and each camera contributes half of every chunk.

**WAFT motion masks (optional, off by default).** The chunk alignment uses
full confidence unless `--with-optical-flow` is passed. When enabled, masks
are computed **per chunk in interleave** with the streaming model (never over
the whole segment — that would OOM): for each chunk, WAFT computes the flow
of the chunk's steps (pairs `(t, t + stride)`, one mask per camera) and the
masks ground the chunk-to-chunk alignment. Masks are **cached by absolute
step index**, so steps shared between overlapping chunks (and the padded
copies of a short final chunk) reuse the previous chunk's masks instead of
re-running optical flow. As with disk-based masks, moving pixels (flow
magnitude > threshold) have their confidence zeroed **during chunk alignment
only** — depth outputs are unaffected. Because the last `stride` frames of a
segment have no pair, they are not streamed when WAFT is on (without it they
are).

## Usage

### 1. All sub-tasks of one episode, single camera, VGGT-Omega

```bash
python tools/astribot/run_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Default camera is the first non-depth camera key (`cam_head` on this
dataset). WAFT is **off by default**; add `--with-optical-flow` to run it
(`weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine`).

### 2. Stereo pair with WAFT motion masks

```bash
python tools/astribot/run_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0 \
    --camera-idxes 4 5 --backend vggt_omega --with-optical-flow
```

Cameras 4/5 are the head stereo pair (`cam_head_stereo_left/right`): both
views run in the same 64-view chunks (32 steps × 2 cameras) and produce
synchronized `depth_cam_head_stereo_left/` + `depth_cam_head_stereo_right/`
folders with identical frame stems.

### 3. DA3 backend, one episode per task

```bash
python tools/astribot/run_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --one-per-task --backend da3
```

## Output

```
<out-dir>/pipeline/                     # <out-dir> defaults to <data-root>/eps_data
└── ep000000/
    ├── subtask_00/
    │   ├── depth_cam_head/             # or depth_cam_head_stereo_left/ + _right/ …
    │   │   ├── frame_000000.npz        # absolute dataset indices, stride-subsampled
    │   │   ├── frame_000004.npz
    │   │   └── …
    │   └── timings.json
    ├── subtask_01/
    └── …
```

Each `frame_<idx>.npz` carries `depth` (H, W, metric), `extrinsics`
(3×4, world→camera), `intrinsics` (3×3) — the same contract as
`run_stream.py`'s output, consumable by `visualize_stream.py` and
`infer_tapip3d.py`. `timings.json` holds `total_s`, `num_chunks` and
`chunk_times_s`.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root passed to LeRobotDataset; full chunk path for chunked datasets |
| `--camera-idxes`, `-c` | first non-depth camera | indices into the dataset's `camera_keys`; `chunk_size % n == 0` required |
| `--episode-idxes`, `-e` | all | only process these episode indices |
| `--one-per-task` | off | select the first episode of each task (exclusive with `-e`) |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root; results under `<out-dir>/pipeline/<episode>/subtask_XX/` |
| `--backend` | `vggt_omega` | streaming backend: `vggt_omega` or `da3` |
| `--stride` | 4 | subsample every N-th frame; with WAFT also the flow pair gap |
| `--with-optical-flow` | off | enable WAFT optical-flow motion masks (default: off — full-confidence alignment) |
| `--motion-threshold`, `-thr` | 2.0 | flow-magnitude (pixel displacement) above which a pixel counts as moving |
| `--waft-checkpoint` | `weights/waftv2/…tf32.engine` | WAFT checkpoint; backend inferred from `.engine`/`.onnx` |
| `--model-path` | backend default | VGGT-Omega artifact override (`.pt2`) |
| `--anyview-model-path` / `--metric-model-path` | backend default | DA3 any-view / metric-depth `.pt2` overrides |
| `--config` | `src/depth_models/streaming/configs/base_config.yaml` | alignment library/method, loop-closure settings |
| `--device` | auto | `cuda` or `cpu` |
| `--skip-done` | off | skip sub-tasks whose `depth_*` outputs already exist |

## Visualization

`tools/astribot/visualize_subtask_stream.py` renders one trajectory video per
sub-task segment from the saved pipeline outputs — the visualization
counterpart of `run_subtask_stream.py`, built on
`tools/general_test/visualize_stream.py` but with the colour frames decoded
**online from the dataset** (no extracted frames or videos on disk):

```bash
python tools/astribot/visualize_subtask_stream.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0 \
    --camera-idxes 4 5 --fps 30        # same selection flags as the stream tool
```

- Selection flags (`-id`, `-d`, `-c`, `-e`, `--one-per-task`, `-x`, `-o`)
  match `run_subtask_stream.py`; cameras must match the ones used for
  streaming. `--stride` only affects the view fitting (the rendered steps are
  the npz stems actually saved).
- Per segment: `--fps` (10), `--size` (960x540), `--max-points` (100000);
  output `<seg_dir>/trajectory.mp4` — each frame shows that step's coloured
  point cloud, camera frustums and the growing camera path, with the view
  fixed per segment (aligned to the first camera, fitted to the union of the
  segment's clouds).
- Segments without npz results are skipped with a message.

## Notes

- **Verified on `astri_making_coffee` (ep000000, 6 sub-tasks, 1744 frames,**
  **stride 4, VGGT-Omega):** 82–105 steps per segment, 2–4 chunks each, all
  npz + timings.json written. Stereo (cameras 4+5) produces identical stem
  sets in both camera folders with distinct extrinsics (right camera offset
  by the ~0.088 m stereo baseline) and distinct depth maps.
- **Mask effect on alignment:** full-confidence alignment uses all
  19,660,800 chunk points; with WAFT masks the static-only point set is
  ~18.4 M and the mean SIM3 alignment error drops roughly 40%
  (0.04–0.06 → 0.02–0.03 on this episode) — moving pixels are excluded from
  the long-trajectory camera alignment only.
- The final `AttributeError: '_thread.RLock' …` printed at interpreter
  shutdown is harmless `multiprocess.resource_tracker` noise (exit code 0).
