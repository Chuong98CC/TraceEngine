# Per-Sub-Task 3D Point Tracking (`tools/astribot/run_step4_traces.py`)

README pipeline **Step 4 (3D trace)** on the dataset, online: for every
sub-task of the selected episodes it tracks the 3D positions of the Step-3
keypoints (SAM3 masks + RoMAv2 init points) with the **TAPIP3D torch.export
programs**, over the Step-2 depth + pose outputs. The RGB frames are
decoded **online** from the LeRobotDataset (one window at a time, nothing
extracted to disk); the geometry comes from the saved
`depth_pose/<episode>/subtask_XX/` folders of `run_step2_depth_stream.py`.
The tracking counterpart of `tools/general_test/module/infer_tapip3d.py`:

    Step-2 stems + Step-3 keypoints (per prompt)
               │
               ▼  one TAPIP3D pass per role (object / manipulator)
    ┌──────────────────────────────┐
    │  coords + visibs per prompt  │
    └──────────────────────────────┘

## How it works

**Two roles, two passes.** The sub-task's prompts are split by role —
matching the prompt text to the [object, manipulator] entries of the
sub-task's row in the dataset's `meta/subtasks.csv` (the same source Step
3a/3b use). Each role is tracked in its **own independent TAPIP3D pass**.
Step 3b samples the **object** keypoints only between the sub-task's 2nd
and 2nd-to-last key-frame (the gripper close/open pair that carries the
object), and each pass traces the Step-2 stems inside its prompts'
key-frame envelope:

| Role | Step-3 key-frames | Tracked sequence (Step-2 stems) |
|---|---|---|
| **manipulator** (e.g. the gripper) | all (sub-task start → last frame) | sub-task's first stem → last stem |
| **object** (e.g. the cup) | close → open (2nd → 2nd-to-last) | last stem ≤ close → first stem ≥ open |

The window rule is role-agnostic (`span_stems` in `utils/keyframe_utils.py`):
from the last stem at-or-before the earliest first key-frame to the first
stem at-or-after the latest last key-frame. The object therefore starts on
the stem **right before the close key-frame** — the object is static until
the close, so its close-frame pixels are exact there (and the gripper has
not occluded them yet) — and ends on the stem **right after the open**.

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (no mask on the frame ->
unconstrained) with **valid depth** at its pixel. When no key-frame of the
sub-task yields any usable point for a role, the role is skipped with the
reason recorded (prompt `metadata.json`, `status: "empty"`). Prompts whose
text matches neither annotation column are tracked too (warned), in a
separate unlabelled pass over their own key-frames.

**Query budget (exact-N).** The shipped iteration program
(`weights/tapip3d/tapip3d_iteration_1088_bf16.pt2`) has a **fixed query
count (1088)**: per pass, up to **64 role keypoints** (Step-3 rank order;
rows trimmed when more) + the full-frame **32x32 support grid** at the
anchor frame. Since the grid's valid-depth points rarely fill the exact
remainder, the support slots are trimmed/padded deterministically to reach
1088: padding points are sampled uniformly at random among the anchor
frame's valid-depth pixels (`np.random.default_rng(seed + role_index)`,
role_index 0 = object, 1 = manipulator). All queries carry **home frame 0**
— the leading stems before the anchor are dropped, not encoded, so the
anchor is always the sequence's first frame (the static iteration graph
cannot mask late-frame queries).

**Online frames + saved geometry.** Each window batch (16 frames) decodes
its RGB frames from the dataset at the stems' absolute indices and reads
the Step-2 geometry (`.lz4` depth + `.npz` pose) of the same stems; both
are resized to the encoder resolution (480x640) with the intrinsics scaled
— the exact math of `load_resized_batch`
(`utils/streaming_utils.py`, split into a shared
`resize_batch_to_inference` helper so the online builder reuses it). The
anchor queries are unprojected with the anchor stem's saved depth + pose.

**Camera and split alignment.** The tracked cameras of a sub-task are
the ones Step 3b wrote init-points subtrees for
(`init_points/.../subtask_XX/<camera>/`): each camera is tracked
separately over its own geometry folder (`depth_pose/.../subtask_XX/
depth_<cam>` — run `run_step2_depth_stream.py` for that camera; a
stereo-only Step-2 run does not cover a mono Step-3 camera). A pass
traces only Step-2
stems (geometry exists there). Its window is the stems inside its
prompts' key-frame envelope — the last stem at-or-before the earliest
first key-frame through the first stem at-or-after the latest last
key-frame (`span_stems`) — and the anchor is the window's first stem
where a prompt has usable keypoints. If Step 2 is later modified to
stream other per-sub-task frame ranges (e.g. at the key-frame indices),
this tool follows the saved stems automatically.

## Usage

### 1. Track every sub-task of episode 0

Requires the Step-2 (`run_step2_depth_stream.py`) and Step-3
(`run_step3_init_points.py`) results of the tracked cameras on disk
(each camera's init points need its own `depth_<camera>` Step-2
outputs):

```bash
python tools/astribot/run_step4_traces.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Each sub-task runs up to two passes (object, then manipulator); the
TAPIP3D encoder/iteration graphs load once per run.

## Visualization

The visualization counterpart of this tool —
`tools/astribot/visualize_step4_traces.py` — renders, per tracked
camera of each sub-task, two videos from the Step-4 trace data:

- `trace2d.mp4` — the world-space keypoint traces projected back onto
  the RGB frames, coloured per role (a 2D overlay of the keypoints with
  their trails);
- `trace3d.mp4` — a 3D point-cloud scene with the growing world-space
  trace curves.

The RGB frames are decoded online from the dataset (nothing extracted to
disk); the videos are written under the shared visualization tree —
`<out-dir>/visualization/<episode>/subtask_XX/<camera>/` — next to the
step-2 `depth_pose.mp4` videos (`visualize_step2_depth_pose.py`, see
`astribot_visualize_step2_depth_pose.md`):

- Standalone and disk-derived: the episodes, sub-task segments, cameras
  and rendered steps are discovered from the saved Step-4 trace outputs
  themselves — no selection flag has to match the `run_step4_traces.py`
  run; `-e` (default: every episode with trace outputs) and `-c`
  (default: every camera with trace outputs) only filter what exists.
  The tool still needs the same Step-2 `depth_pose` outputs as geometry
  for `trace3d.mp4`.

```bash
python tools/astribot/visualize_step4_traces.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Instead of the videos, `--render stills` saves one trace2d PNG per
Step-3 key-frame of the camera — the same 2D overlay (role-colored
keypoints + trails, occluded red, HUD + legend) composited on the
key-frame image itself, read off the Step-3 `key_frames/` tree (no
online frame decode), under
`<out-dir>/visualization/<episode>/subtask_XX/<camera>/trace2d_stills/frame_<k>.png`.
The Step-3 key-frames sit at arbitrary dataset indices while the trace
rows live on the strided Step-2 stems, so each key-frame draws the trace
state of the stem nearest to it in time (the HUD reads
`kf <k> ~ stem <s>`); prompts without a row there (e.g. the object
before its transport window) stay undrawn, exactly as in the video:

```bash
python tools/astribot/visualize_step4_traces.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0 --render stills
# --keyframes-root overrides the default <out-dir>/sampling_points/key_frames
```

## Output

```
<out-dir>/traces/                    # <out-dir> defaults to <data-root>/eps_data
└── ep000000/
    └── subtask_00/
        └── cam_head/                # one camera of the sub-task (its own
                                     #   Step-3 init points + depth_<cam> outputs)
            ├── metadata.json        # roles/passes, prompt statuses, skip reasons
            ├── brown_cup/           # object pass, per-prompt slice
            │   ├── coords.npy       # (T, Q, 3) world-space 3D traces
            │   ├── visibs.npy       # (T, Q) visibility flags (sigmoid >= threshold)
            │   ├── queries.npy      # (Q, 4) query points (home frame 0, x, y, z)
            │   └── metadata.json    # role, anchor, steps, query/keypoint mapping
            └── left_robot_arm_s_grippers/   # manipulator pass
                └── …
```

`T` = the number of tracked stems (the trace window's Step-2 stems from
the anchor on, absolute indices listed in the prompt `metadata.json`
under `steps`), `Q` = the prompt's tracked keypoints (<= 64). The pass arrays
(including the support queries) are sliced to each prompt's own queries;
prompts skipped by a role carry a `metadata.json` with
`status: "empty"` + `empty_reason` only (the Step-3b convention).
`coords` row 0 equals the anchor-world points of `queries.npy`
(refined by the window iterations afterwards).

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--repo-id`, `-id` | — (required) | dataset repo id as seen by LeRobotDataset |
| `--data-root`, `-d` | — (required) | root of the local dataset copy |
| `--camera-idxes`, `-c` | all | dataset camera indices eligible for tracking (the tracked cameras of each sub-task are the ones with Step-3 init points on disk) |
| `--episode-idxes`, `-e` | all | only process these episode indices (default: all episodes with Step-3 init points on disk) |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root; Step-2 results read under `<out-dir>/depth_pose`, traces saved under `<out-dir>/traces`. The Step-3 inputs are always read from the sampling_points root (`<data-root>/eps_data/sampling_points/{detections,init_points}`) |
| `--vis-threshold` | 0.5 | sigmoid visibility threshold for `visibs` |
| `--seed` | 0 | RNG seed for the support padding (per role: seed + 0 object, +1 manipulator, +2 unlabelled) |
| `--skip-done` | off | skip prompts whose `coords.npy` already exists |

The TAPIP3D artifacts and graph config are the shipped defaults (encoder
`weights/tapip3d/tapip3d_encoder_480x640_bf16.pt2`, fused corr+updater
`weights/tapip3d/tapip3d_iteration_1088_bf16.pt2` — 1088 = 32x32 support
grid + 64 object slots, 6 iterations inside each window).

## Notes

- **Roles come from `meta/subtasks.csv`** (Step 3 requires it too): the
  tool reads the same `subtasks.csv` annotations as the Step-3 tools
  (`load_subtask_meta`), so prompt texts must match the [object,
  manipulator] entries (they do when Step 3a/3b recorded them from there).
  The row of a segment is resolved through the canonical label that Step
  3a recorded as `subtask_index` in the camera's detections JSON (the
  segment
  ordinals are not the canonical ids — an episode executes its sub-tasks
  e.g. in order `[0, 2, 1, 3, 5, 4]`); a segment whose JSON carries no
  label is tracked unlabelled (anchored like the object) with a warning —
  never role-matched by the segment ordinal.
- **Anchors sit on the stem grid; boundary key-frames usually don't.**
  Step-2 streams at `--stride` 4, so a prompt's key-frames are generally
  *not* stems — the sub-task's first frame (a boundary key-frame) is a
  stem only when it lands on the grid, and usually it does not. When the
  start key-frame is a stem the manipulator window starts on it;
  otherwise the window starts on the stem nearest the start key-frame —
  the last one at-or-before it, or the first one at-or-after it when the
  sub-task's grid begins later. The manipulator pass therefore anchors on
  the sub-task's first stem as before; the object pass anchors on the
  stem right before the close key-frame, where the object is still static
  (its close-frame pixels are exact there, and the approach has not
  occluded them). When that stem carries no usable keypoints the anchor
  advances to the next window stem that does.
- **Sequences shorter than the 16-frame window** (`seq_len` of the
  exported encoder) run no real window: the output stays at the anchor
  points with all `visibs` false — a warning is printed.
- The TAPIP3D path runs validated numerics (fp32, TF32 off); importing
  `flow_models.tapip3d` disables the flash/mem-efficient SDPA kernels
  globally (irrelevant here — no SAM3 runs in this tool).
- **Verified on `astri_making_coffee` (ep000000, all 6 sub-tasks, camera
  cam_head, Step-2 at stride 4, VGGT-Omega):** 12 passes (object +
  manipulator per sub-task) each tracked 64 role keypoints + 1024 support
  queries (exact 1088) anchored at the sub-task's first stem (0, 330, 467,
  884, 1227, 1372 — every sub-task start was a usable key-frame, so no
  object pass had to advance). Per-prompt `coords (T, 64, 3)` with T =
  35–105 stems; row 0 matches the anchor world points (~1e-3 m, the window
  iterations' refinement) and, projecting row 0 back to pixels with the
  anchor's Step-2 pose, **all 64 queries land inside the Step-3 SAM3 mask**
  of the anchor key-frame (64/64 for all 12 prompts) — the Step-3 pixels →
  world → trace chain is closed. Visibility rates vary with scene-driven
  occlusion (e.g. the cup drops to ~0 % once the gripper closes on it
  around abs frame 32 of subtask_00 — the grasp hides it from the head
  view for the rest of the transfer — while subtasks where the hand
  occludes less show 60–85 %). Re-verified 2026-09-06 after the
  object/manipulator span split (ep000000, all 6 sub-tasks, Step 3a/3b/4d
  re-run): the object passes now trace the close..open transport only —
  anchored on the last stem at-or-before the close key-frame (168, 366,
  611, 1048, 1231, 1448 — each usable directly, so no pass had to advance)
  and ending on the first stem at-or-after the open one (312, 466, 755,
  1168, 1355, 1564; the 466 end is the grid clamp — sub-task 1's open
  key-frame 502 lies beyond its last Step-2 stem) — while the manipulator
  passes still span the sub-task's full [start .. end] key-frame envelope
  on the Step-2 grid (grid-clamped windows: 0..328, 338..466, 555..883,
  900..1200, 1227..1371, 1400..1740).
