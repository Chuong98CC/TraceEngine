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
3a/3b use). Each role is tracked in its **own independent TAPIP3D pass**,
anchored differently:

| Role | Anchor | Tracked sequence |
|---|---|---|
| **manipulator** (e.g. the gripper) | the sub-task's **first frame** (its first Step-2 stem) | first frame → sub-task end |
| **object** (e.g. the cup) | the sub-task's **first key-frame that carries usable object keypoints** | that key-frame → sub-task end (leading stems skipped when the sub-task start has none) |

A keypoint is *usable* on a key-frame when it is a surviving Step-3
keypoint lying inside that key-frame's SAM3 mask (no mask on the frame ->
unconstrained) with **valid depth** at its pixel. When no key-frame of the
sub-task yields any usable point for a role, the role is skipped with the
reason recorded (prompt `metadata.json`, `status: "empty"`). Prompts whose
text matches neither annotation column are tracked too (warned, anchored
like the object).

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

**Camera and split alignment.** The tracked camera is the one Step 3
recorded (`camera_key` in each prompt's `init_points.json`), and the
geometry folder must exist for it (`depth_pose/.../subtask_XX/depth_<cam>`
— run `run_step2_depth_stream.py` for that camera; a stereo-only Step-2
run does not cover a mono Step-3 camera). A key-frame can only anchor a
pass when it is a Step-2 stem (has geometry): the anchor scan runs over
the role's key-frames that are stems, starting from the sub-task's first
stem. The tracked sequence is *whatever Step-2 stems are saved* from the
anchor on — if Step 2 is later modified to stream other per-sub-task
frame ranges (e.g. at the key-frame indices), this tool follows the saved
stems automatically.

## Usage

### 1. Track every sub-task of episode 0

Requires the Step-2 (`run_step2_depth_stream.py`) and Step-3
(`run_step3_init_points.py`) results of the same camera on disk:

```bash
python tools/astribot/run_step4_traces.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee \
    --episode-idxes 0
```

Each sub-task runs up to two passes (object, then manipulator); the
TAPIP3D encoder/iteration graphs load once per run.

## Output

```
<out-dir>/traces/                    # <out-dir> defaults to <data-root>/eps_data
└── ep000000/
    ├── metadata.json                # roles/passes, prompt statuses, skip reasons
    └── subtask_00/
        ├── metadata.json
        ├── brown_cup/               # object pass, per-prompt slice
        │   ├── coords.npy           # (T, Q, 3) world-space 3D traces
        │   ├── visibs.npy           # (T, Q) visibility flags (sigmoid >= threshold)
        │   ├── queries.npy          # (Q, 4) query points (home frame 0, x, y, z)
        │   └── metadata.json        # role, anchor, steps, query/keypoint mapping
        └── left_robot_arm_s_grippers/   # manipulator pass
            └── …
```

`T` = the number of tracked stems (the sub-task's Step-2 steps from the
anchor on, absolute indices listed in the prompt `metadata.json` under
`steps`), `Q` = the prompt's tracked keypoints (<= 64). The pass arrays
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
| `--camera-idxes`, `-c` | all | dataset camera indices eligible for tracking (the per-sub-task camera comes from its Step-3 init points) |
| `--episode-idxes`, `-e` | all | only process these episode indices (mutually exclusive with `--one-per-task`) |
| `--one-per-task` | off | select the first episode of each task |
| `--max-episodes`, `-x` | — | cap the number of processed episodes |
| `--out-dir`, `-o` | `<data-root>/eps_data` | output root; Step-2 results read under `<out-dir>/depth_pose`, Step-3 under `<out-dir>/init_points`, traces under `<out-dir>/traces` |
| `--image-size` | `480 640` | inference resolution (H W), must match the encoder graph |
| `--encoder` | `weights/tapip3d/tapip3d_encoder_480x640_bf16.pt2` | TAPIP3D encoder `.pt2` artifact |
| `--iteration` | `weights/tapip3d/tapip3d_iteration_1088_bf16.pt2` | fused corr+updater `.pt2` (query count auto-detected; 1088 = 32x32 support grid + 64 object slots expected) |
| `--num-iters` | 6 | fused corr+updater iterations inside each window |
| `--vis-threshold` | 0.5 | sigmoid visibility threshold for `visibs` |
| `--seed` | 0 | RNG seed for the support padding (per role: seed + 0 object, +1 manipulator, +2 unlabelled) |
| `--device` | auto | `cuda` or `cpu` (TAPIP3D is GPU-only) |
| `--skip-done` | off | skip prompts whose `coords.npy` already exists |

## Notes

- **Roles come from `meta/subtasks.csv`** (Step 3 requires it too): the
  tool reads the same `subtasks.csv` annotations as the Step-3 tools
  (`load_subtask_meta`), so prompt texts must match the [object,
  manipulator] entries (they do when Step 3a/3b recorded them from there).
  The row of a segment is resolved through the canonical label that Step
  3a recorded as `subtask_index` in the detections JSON (the segment
  ordinals are not the canonical ids — an episode executes its sub-tasks
  e.g. in order `[0, 2, 1, 3, 5, 4]`); a segment whose JSON carries no
  label is tracked unlabelled (anchored like the object) with a warning —
  never role-matched by the segment ordinal.
- **Anchor key-frames must be Step-2 stems.** Step-1 key-frames include
  the sub-task boundary frames, and Step-2 streams from the sub-task's
  first frame at `--stride` 4 — in the current pipeline only the
  sub-task's first frame (a boundary key-frame) is generally a stem, so
  both roles usually anchor there (the object rule only advances when its
  start key-frame carries no usable points but a later stem does).
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
  occludes less show 60–85 %).
