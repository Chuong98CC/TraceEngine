# Astribot output layout + depth_pose storage format

Date: 2026-09-10
Status: approved design, not yet implemented

## 1. Goal

Replace the per-task output roots under `<data_root>/eps_data/` with one
episode-centric tree, `<data_root>/episodes/`, in which every per-sub-task
artifact lives under `<eps>/<subtask>/<task>/<camera>`. At the same time
replace the per-frame `frame_<idx>.lz4` + `frame_<idx>.npz` pair of the
Step-2 `depth_pose` output — 2 files per frame per camera, ~150k–250k files
for the current 62-episode corpus — with one indexed depth container plus one
consolidated pose file per (subtask, camera).

Two problems drive this:

- **File count.** A processed segment writes 2 files per frame per camera
  (e.g. 85 frames -> 170 files for one camera). Directory walks, `du`,
  rsync/upload, and any full-tree scan pay that in inodes and metadata ops
  rather than in bytes: the 669-byte pose `.npz` is *smaller than its own
  zip overhead* and outnumbers the depth payloads one-for-one.
- **Pose is trapped inside the depth file pair.** Pose-only consumers
  (`visualize_stream.load_trajectory`, `run_step4_traces._pose_at`,
  `visualize_tapip3d`) must open a per-frame `.npz`, read `shape`, then
  decode the full depth map just to throw the depth away.

Measured on one real segment (85 frames, `ep000000/subtask_02/depth_cam_head`,
7.1 MB of `.lz4`): warm-cache decompression of all 85 frames is ~15 ms, so
raw throughput is not the bottleneck — file granularity is. A 62-episode run
at stride 4 and 2–3 cameras lands at roughly 150k–250k files under
`depth_pose/` alone.

## 2. Layout

```
<data_root>/episodes/
├── ep000/
│   ├── subtask.json                      # Step 1: splits + labels, one file
│   ├── split_graph.png                   # Step 1: gripper plot
│   └── subtask_00/
│       ├── frames/                       # extract_frames --mode frames
│       │   ├── cam_head/frame_000556.jpg
│       │   └── depth_cam_head/frame_000556.lz4
│       ├── videos/                       # extract_frames --mode videos
│       │   └── cam_head/video.mp4
│       ├── depth_pose/                   # Step 2
│       │   └── cam_head/{depth.lz4, poses.npz}
│       ├── sampling_points/              # Step 3
│       │   ├── key_frames/cam_head/frame_000556.jpg
│       │   ├── detections/cam_head.json
│       │   └── init_points/{cam_head/..., metadata.json}
│       ├── traces/cam_head/<prompt_slug>/{coords.npy, visibs.npy,
│       │                                  queries.npy, metadata.json}
│       └── visualization/cam_head/{depth_pose.mp4, trace2d.mp4, trace3d.mp4}
└── ep001/…
```

Rules:

- **Episode dir**: `ep{ep_idx:03d}`; **sub-task dir**: `subtask_{k:02d}`
  (the sub-task name is unchanged; the episode prefix shortens from six
  digits to three, which still sorts correctly — the corpus holds 62
  episodes, so 999 is not a constraint).
- **Per-sub-task task dirs**: `frames`, `videos`, `depth_pose`,
  `sampling_points`, `traces`, `visualization`.
- **Camera subdir**: the camera name as today (`cam_head`,
  `cam_head_stereo_left`, …). Inside `frames/` the depth folder keeps the
  `depth_<camera>` prefix (`frames/depth_cam_head/`) because those files are
  per-frame `.lz4`, not containers. Inside `depth_pose/` the camera dir is
  bare (`depth_pose/cam_head/`) since the task dir already says what it is.
- **Episode-level files** are the only exception to the pattern:
  `subtask.json` and `split_graph.png` sit directly in the episode dir
  (the gripper plot is one per episode, so it carries no task-id suffix).
  There are no episode-level task dirs.
- `frames/` keeps per-frame files (`.jpg` + `.lz4`): those are RGB/depth
  *inputs* addressed by frame name/index, and general_test tools read them
  as images. Only `depth_pose` changes format.

### 2.1 Path module

New `src/utils/astribot_paths.py` is the single source of truth. Every tool
calls it instead of joining `"eps_data"/"depth_pose"/...` itself (today the
`frame_%06d` name is hand-built in 6+ places):

```python
EPISODES_DIR = "episodes"
TASKS = ("frames", "videos", "depth_pose", "sampling_points", "traces", "visualization")

episodes_root(data_root, out_dir=None) -> Path      # <data_root>/episodes, or out_dir
episode_dir(root, ep_idx) -> Path                   # <root>/ep000
subtask_json(root, ep_idx) -> Path                  # <root>/ep000/subtask.json
split_graph(root, ep_idx) -> Path                   # <root>/ep000/split_graph.png
subtask_dir(root, ep_idx, subtask_k) -> Path        # <root>/ep000/subtask_00
task_dir(root, ep_idx, subtask_k, task, camera=None) -> Path
discover_episodes(root) -> list[int]                # for the visualizers
discover_subtasks(root, ep_idx) -> list[int]
discover_cameras(root, ep_idx, subtask_k, task) -> list[str]
frame_stem(idx) -> str                              # "frame_%06d" (frames/ only)
```

Episodes are discovered by `subtask_{k:02d}` subdirs (or `subtask.json`),
not by task dirs, so a partially processed episode is still discoverable.
Every `^ep(\d{6})$` / `ep%06d` / `ep{f"{i:06d}"}` in discovery, writing and
docs becomes the three-digit form (notably `keyframe_utils._EP_RE`, the
`startswith("ep")` scans in the visualizers, and
`run_step2_depth_stream.py`'s segment dir).

CLI shape is unchanged: `--data-root` stays required, `--out-dir` still
overrides the root and now defaults to `<data-root>/episodes` instead of
`<data-root>/eps_data`.

## 3. Step 1 (`tools/astribot/extract_frames.py`)

`subtask_splits.json` (from `detect_subtask`) and `subtask_labels.json`
(from `key_frames`) merge into **one** `episodes/ep{ep:03d}/subtask.json`:

```json
{
  "episode": 0,
  "task_id": 0,
  "task": "Make coffee",
  "from_idx": 0,
  "to_idx": 1744,
  "key_frames": [0, 169, 311, ...],
  "split_frames": [339, 557, 901, 1199, 1401],
  "labels": [0, 2, 1]
}
```

`labels` is a plain list indexed by segment ordinal holding the canonical
subtask label, `null` for a segment beyond the ground-truth order. This drops
today's `{"source": ..., "segments": {...}}` wrapper: the list is the whole
payload, and the resolution source (`ground_truth` / `annotations` /
`ordinal`) is only a diagnostic — it stays in the `[labels] …` stdout line
that `_label_keyframe_segments` already prints, not in the file.

**Ownership moves:** the label resolution (`_label_keyframe_segments`) runs in
`detect_subtask`, not `key_frames`. The label order comes from the dataset
annotations (the frame table's `subtask_index` column, else
`meta/lerobot_annotations.json`), which is independent of key-frame
extraction — `key_frames` was only ever the file that happened to write it.
`detect_subtask` already loads the episode frame table, so it has everything
it needs; `key_frames` drops the call.

Mode outputs (MODE_ROOTS retires in favour of `astribot_paths`):

| mode | old | new |
| --- | --- | --- |
| `detect_subtask` | `subtask/ep{ep}/subtask_splits.json` + `split_graph_*.png` | `ep{ep}/subtask.json` + `split_graph.png` |
| `key_frames` | `key_frames/ep{ep}/subtask_XX/<cam>/frame_*.jpg` + `subtask_labels.json` | `ep{ep}/subtask_XX/sampling_points/key_frames/<cam>/frame_*.jpg` (no labels) |
| `videos` | `subtask_videos/ep{ep}/<cam>/subtask_XX.mp4` | `ep{ep}/subtask_XX/videos/<cam>/video.mp4` |
| `frames` | `subtask_frames/ep{ep}/subtask_XX/<cam>/` + `depth_<cam>/` | `ep{ep}/subtask_XX/frames/<cam>/` + `depth_<cam>/` |

The reading modes (`key_frames`, `videos`, `frames`) keep reading `split_frames`
from `subtask.json`, with the same fallback as today: when the chosen output
root has no `subtask.json` for the episode, read the canonical
`<data-root>/episodes/ep{ep:03d}/subtask.json`.

`src/utils/keyframe_utils.py` loses `subtask_labels_path` / `SUBTASK_LABELS_FILE`
and gains `load_subtask(root, ep_idx)` returning the merged dict; the old
`load_subtask_labels(root, ep_idx)` survives as a thin accessor returning the
`labels` list (`list[int | None]`, index = segment ordinal) so Step 3a's call
sites keep working with `labels[segment]` instead of `data["segments"][str(segment)]`.
Its `sampling_points_root` / `keyframes_root` / `episode_dir` / `keyframe_path`
helpers delegate to `astribot_paths`.

## 4. depth_pose format

Two files per (subtask, camera), replacing 2 files per frame:

```
depth_pose/cam_head/
├── depth.lz4      # one appendable, indexed container
└── poses.npz      # every frame's pose, one file
```

### 4.1 `depth.lz4` container

```
[header 64B]
  magic          b"DPK1"        (4)
  version        uint32 LE = 1  (4)
  height, width  uint32 LE      (8)
  codec          uint32 LE = 0  (4)   0 = LogDepthToUint8 over [min_depth, max_depth]
  min_depth      float32 LE     (4)
  max_depth      float32 LE     (4)
  frame_count    uint64 LE      (8)   0 until finalized
  index_offset   uint64 LE      (8)   0 until finalized
  reserved                      (20)
[frame 0]        uint64 LE payload_len + payload   (the lz4 bytes save_depth_lz4
[frame 1]        …                                  produces today, unchanged)
[index]          uint64 LE offsets[frame_count+1]  (written at close; byte offset
                                                    of each payload start, plus a
                                                    terminal end offset)
[trailer 16B]    uint64 LE index_offset + b"DPK1IDX"
```

- **Sequential read** (the common case: every consumer scans all frames): one
  `open`, read the header, then read length-prefixed records front to back —
  no index needed.
- **Random access**: read the trailer from EOF, seek to `index_offset`, load
  `offsets`, then one seek + one `lz4.frame.decompress` per frame.
- **Unfinalized file** (a run killed mid-segment): header and record framing
  are still valid, so the reader can scan sequentially and recover the frames
  that made it. `poses.npz` is absent, which is the completion marker.
- The codec is recorded in the header, so the container is self-describing;
  decoding uses `LogDepthToUint8Transform` with the header's range, and the
  stored payload is metres after decoding, exactly as today.

### 4.2 `poses.npz`

Written with `np.savez` (uncompressed, so it stays mmap-able — poses are tiny
and deflate was pure overhead at 669 B/file):

| key | shape | dtype |
| --- | --- | --- |
| `frame_indices` | (T,) | int64 — absolute dataset frame indices (the number in `frame_%06d`) |
| `extrinsics` | (T, 3, 4) | float32 — world-to-camera, SIM3-warped into chunk 0 |
| `intrinsics` | (T, 3, 3) | float32 |
| `shape` | (2,) | int64 — (H, W) of the depth maps (kept: readers validate against it) |

**This retires the `frame_%06d` stem as the interface between producer and
consumer:** frame identity is `frame_indices`, and readers address frames by
absolute index, not by filename. Consumers that need a "sorted stem list"
(`run_step4_traces.seg_stems`, `visualize_step2_depth_pose` stems,
`visualize_stream.load_stems`) now sort `frame_indices` instead — same
ordering guarantee, without depending on zero-padding width.

### 4.3 Writer / reader API

New `src/utils/depth_pose_io.py`:

```python
class DepthContainerWriter:            # one camera's depth
    def __init__(self, path, height, width, clip=(MIN_DEPTH, MAX_DEPTH))
    def append(self, depth_m) -> None  # encodes + lz4 + appends, streaming
    def close(self) -> None            # index + trailer + header patch

class DepthContainerReader:
    frame_count, shape, clip -> ...
    def read(self, i) -> np.ndarray    # float32 metres, position i
    def read_range(self, a, b)         # sequential slice without index seeks

class DepthPoseWriter:                 # one (subtask, camera) dir
    def __init__(self, out_dir, height, width)
    def add(self, frame_index, depth_m, extrinsics, intrinsics) -> None
    def close(self) -> None            # finalize depth.lz4 + write poses.npz

class DepthPoseReader:
    frame_indices -> np.ndarray
    def depth(self, i) -> np.ndarray   # by position
    def depth_at(self, frame_index)
    def pose_at(self, frame_index) -> (extrinsics, intrinsics)   # no depth read
    shape -> (H, W)
```

`DepthPoseReader.pose_at` is what makes the pose-only consumers cheap: it
touches `poses.npz` and never opens `depth.lz4`.

### 4.4 Step-2 write path

`BaseStreaming.run()` keeps its two-pass structure (chunk loop writes depth,
final pass applies the cumulative SIM3 scale), because the scale of chunk *k*
is only known after chunk *k+1* is processed:

1. Chunk loop: `DepthContainerWriter.append` per frame per view — appended in
   order, streaming, no buffering of the segment.
2. Final pass: for each (subtask, camera) a replacement container is streamed
   to `depth.lz4.tmp` with the per-frame scale applied (`reader.read(i) * s`),
   then `os.replace` onto `depth.lz4`; `poses.npz` is written last from the
   in-memory `all_extrinsics`/`frame_meta` lists (already buffered for the
   whole run today).

Stored depth therefore stays **metres, matching today's semantics exactly**
(including the re-encode clipping at `MAX_DEPTH`), and one sequential
read+rewrite replaces today's N open-modify-close cycles.

*Rejected alternative:* store raw model-scale depth plus a per-frame `scale`
array in `poses.npz` and multiply on read. It deletes the second pass but
changes clipping behaviour (today the rescaled values are clipped when
re-encoded) and makes silently-unscaled depth possible for any consumer that
bypasses the reader.

`--skip-done` (`run_step2_depth_stream.py`) becomes "`poses.npz` exists for
every selected camera", which is also the completion marker everywhere else.

## 5. Consumers

The shared layer does most of the work; everything that goes through it is
migrated by changing four functions.

**`src/utils/streaming_utils.py`** (single choke point):

| function | change |
| --- | --- |
| `load_stream_data(img_dir, result_dir, stem)` | take the `depth_pose/<cam>` dir + absolute frame index; return metres, ext (3,4), intr |
| `load_pair(...)` | same, plus the RGB path from `frames/<cam>/` |
| `load_npz_batch(...)`, `load_resized_batch(...)` | drop the `frame_%06d` join; index into `DepthPoseReader` |
| `compute_global_depth_roi(...)` | sequential container scan |
| `scan_image_folder(...)` | unchanged (it scans `frames/` jpgs) |

**Direct readers to update:**

- `tools/astribot/run_step4_traces.py` — `seg_depth_dir` via `astribot_paths`;
  `seg_stems` from `reader.frame_indices`; `_geometry_at` via
  `reader.depth_at`. The window/snap logic in `keyframe_utils.span_stems`
  keeps working on frame-index lists unchanged.
- `tools/astribot/visualize_step2_depth_pose.py` — episode/subtask/camera
  discovery via `astribot_paths`; stems from `frame_indices`.
- `tools/astribot/visualize_step4_traces.py` — `_pose_at` becomes a real
  pose-only read (`pose_at`), no depth decode.
- `tools/general_test/pipeline/visualize_stream.py` — `load_stems` from
  `poses.npz`; `load_trajectory` pose-only; `_union_scene_points` and the
  per-step loaders via the reader.
- `tools/general_test/pipeline/visualize_rgbd.py` — folder mode reads
  `poses.npz` + container instead of asserting per-frame npz keys.
- `tools/general_test/module/infer_tapip3d.py` — `--depth-dir` now points at a
  `depth_pose/<cam>` dir (the shared helpers do the rest).
- `src/utils/visualize/visualize_tapip3d.py` — pose reads via `pose_at`.

**Writers and path plumbing:**

- `src/depth_models/streaming/base_streaming.py` — `prepare()` takes camera
  names and a depth_pose dir; the save loop and final pass move to
  `DepthPoseWriter`/`DepthContainerWriter`.
  - **Consequence:** `BaseStreaming` is shared with
    `tools/general_test/pipeline/run_depth_stream.py`, so its `./exps/...`
    outputs become containers too. One format everywhere, no compat branch.
- `tools/astribot/run_step2_depth_stream.py` — `seg_dir` =
  `.../subtask_XX/depth_pose`; `--skip-done` checks `poses.npz`. The
  per-segment `timings.json` goes away: `_report_run_stats` (in
  `tools/general_test/pipeline/run_depth_stream.py`, shared by both callers)
  keeps printing its timing/memory summary but stops writing the file, so
  folder-mode streaming under `tools/general_test/` drops it too. Nothing
  reads the file — only the docs reference it.
- `tools/astribot/extract_frames.py` — section 3.
- `tools/astribot/run_step4_traces.py` — drops the **camera-level**
  `traces/<cam>/metadata.json` roll-up (`prompts[]`/`passes[]`/`depth_dir`
  summary, written at :694): nothing in the repo reads it. The **per-prompt**
  `metadata.json` (written at :910 for tracked prompts, :931 for skipped
  ones) stays — `visualize_step4_traces.py` needs its `status` to decide
  whether a prompt is renderable and its `steps` to map `coords.npy` rows to
  absolute dataset frames, and `prompt`/`role` to label the render. Step 4
  itself never reads either file (its own input is Step-3b's
  `init_points.json`).
- `tools/astribot/run_step3_motion_masks.py`,
  `tools/astribot/run_step3_init_points.py`,
  `tools/general_test/pipeline/run_object_detection.py` (per-sub-task
  `detections/<cam>.json`; the JSON already groups by sub-task, so it drops
  one nesting level), `tools/general_test/pipeline/run_object_init_points.py`,
  `src/utils/keyframe_utils.py`.

**Scripts and docs:** the 7 scripts under `scripts/astribot/`, the 3
general_test wrappers pointing at `eps_data/subtask_frames/…`, and
`docs/astribot/{astribot_extract_frames,astribot_subtask_stream,astribot_traces,astribot_visualize_step2_depth_pose}.md`
+ the affected `docs/general_test/` pages (`pipeline/step2.md`,
`module/depth_streaming.md` and any other page that names `timings.json` or
the old `depth_pose` file layout). `astribot_traces.md` also loses the
camera-level `metadata.json` from its output tree (its per-prompt entry
stays).

## 6. Migration

Per decision: **no converter, no dual-read compatibility.** `eps_data/` is
regenerated into `episodes/` by re-running Steps 1–4. Nothing in the code
reads `eps_data` after this change; the shell scripts are repointed.

## 7. Verification

1. `tests/test_depth_pose_io.py` (CPU, always run): container round-trip
   (write T frames -> read all, random access, `read_range`), unfinalized
   container recovery, `poses.npz` round-trip, empty/short segment,
   `frame_indices` ordering, path-module layout invariants.
2. **Golden equivalence on real data** (pytest, skipped when the corpus is
   absent): read an existing `eps_data/depth_pose/.../depth_cam_head/`
   segment, pile the per-frame `.lz4` payloads into a container via
   `DepthContainerWriter`, and assert every frame decodes bit-identical to
   `load_depth_lz4` on the original file, and that `poses.npz` matches the
   per-frame `.npz` values. This is the guard against a silent
   precision/scale regression and needs no GPU.
3. End-to-end smoke on one small segment: Step 2 -> Step 3a/3b -> Step 4 ->
   both visualizers, on the new tree; the visualizers are the cheapest way to
   confirm depth/pose still line up visually.
4. `uv run --extra dev pytest tests/ -q` green.

## 8. Out of scope

- `src/mu0/` — it reads the packed TraceExtract layout (`images.npy`,
  `depth.npy`), never `depth_pose`.
- The legacy `eps_data/depth_pose_uint16/` tree
  (`tests/depth_quantize_error.py`).
- Per-frame files inside `frames/` (inputs addressed by frame name).
- `docs/superpowers/plans/*` historical documents keep their `eps_data/`
  references.

## 9. Risks

- **Silent scale regression** in the two-pass rewrite: covered by the golden
  equivalence test (§7.2) plus the visualizer smoke test.
- **`frame_indices` as the new contract**: any consumer that assumed
  contiguous or globally-strided stems must keep using the list, not
  arithmetic. The existing `_snap_stems`/`span_stems` logic already treats
  stems as a list; the refactor must not introduce index arithmetic.
- **Partially migrated trees**: an episode mid-run has `depth.lz4` without
  `poses.npz`; every reader and `--skip-done` treats that as incomplete.
