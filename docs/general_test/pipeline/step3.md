# Pipeline Test · Step 3 — Sampling Keypoints

End-to-end test of **Step 3 (Sampling Keypoints)** of the repo README: for
each interacted object, detect the object on the sub-task's key-frames,
segment its masks, match keypoints across the key-frames, and keep the
top-k keypoints inside the masks. In dataset mode, object prompts are
matched between the sub-task's 2nd and 2nd-to-last key-frame (the gripper
close/open pair); manipulator prompts — and all folder-mode prompts, which
carry no role — over all key-frames.

Step 3 runs on the **key-frames that Step 1 saved to disk**
(`tools/astribot/extract_frames.py --mode key_frames`) — a handful of frames
per sub-task, so they are persisted instead of decoding the episode videos
online (unlike Step 2, which streams every frame). The on-disk layout is:

```
<keyframes_root>/ep{ep:06d}/subtask_{k:02d}/<camera>/frame_<idx:06d>.jpg
```

In dataset mode the Step-3 driver (`run_step3_init_points.py`, and the
standalone 3a/3b episode runs) defaults its output root to
`<data-root>/eps_data/sampling_points` — `key_frames/`, `detections/` and
`init_points/` under it (`--out-dir` to change) — while Step 2's
`depth_pose/`, Step 4's `traces/` and the shared detect_subtask `subtask/`
splits stay under `<data-root>/eps_data` (Step 1a of the driver writes the
splits there, never under the sampling root).

## What the step does

```
text prompt per interacted object — dataset mode: the manipulator/object
columns of the sub-task's row in meta/subtasks.csv; folder mode: --text-prompts
  → RexOmni     — detect the object in each key-frame            (Step 3a)
  → SAM3        — segment the object masks (bbox + text prompt)  (Step 3b)
  → RoMAv2      — match keypoints across key-frames on enlarged
                  bbox crops (by default the mask is cropped with
                  the same box and fed to RoMAv2, so candidates
                  are sampled inside the object only;
                  --sampling-mode uniform samples over the whole
                  crops instead and the in-mask top-k filter
                  decides — same crops and filter, different pool;
                  dataset mode: object prompts span the close..open
                  key-frames)
  → top-k keypoints inside the object masks
```

In dataset mode the prompts are **per sub-task** (recorded next to each
sub-task's detections in the Step-3a JSON — there is no global prompt list),
and the dataset must carry the `meta/subtasks.csv` annotations. A sub-task's
row is found through its **canonical label**, not its segment ordinal:
Step 1 (`key_frames`) matches every segment to the ground-truth execution
order and writes `subtask_labels.json` next to the key-frames (the
canonical ids can run e.g. `[0, 2, 1, 3, 5, 4]` while the segments are
numbered `subtask_00…`), and Step 3a reads that file — a key-frame
extraction without it is refused (no silent ordinal fallback). See
[`astribot_extract_frames.md`](../../astribot/astribot_extract_frames.md)
for the label resolution.

Two sub-steps, two environments — they **cannot share a process**:

| Sub-step | Tool | What it runs | Environment |
|---|---|---|---|
| 3a | `tools/general_test/pipeline/run_object_detection.py` | RexOmni open-vocabulary detection → one JSON per (episode, camera) | `.venv-rexomni` (Python 3.10 / torch 2.7, see [`rexomni.md`](../module/rexomni.md)) |
| 3b | `tools/general_test/pipeline/run_object_init_points.py` | SAM3 masks + RoMAv2 keypoints → `init_points/` per camera + prompt | main env (`.venv`, `.pt2` runtimes) |

The high-level drivers launch the two tools **sequentially as
subprocesses**, so you run them from the **main** environment:

```
run_e2e_init_points.py / run_step3_init_points.py  (drivers)
    ├── 3a: run_object_detection.py  →  <out>/detections/ep{idx:06d}/<camera>.json
    └── 3b: run_object_init_points.py →  <out>/init_points/ep{idx:06d}/subtask_XX/<camera>/<prompt_slug>/
```

## Entry points

| Entry point | Scope | What it runs |
|---|---|---|
| `tools/general_test/pipeline/run_e2e_init_points.py` | **one key-frame folder** (one sub-task of one camera, dataset-independent) | 3a + 3b; the folder is sub-task 00 of a synthetic episode labelled `--episode-idx` (default 0), the folder name is the camera key |
| `scripts/general_test/infer_step3.sh` | wrapper of `run_e2e_init_points.py` | `python tools/general_test/pipeline/run_e2e_init_points.py --keyframes-dir "$1" "$@"` |
| `tools/astribot/run_step3_init_points.py` | **episodes on the dataset** | Step 1 (extract the sub-task key-frames) + 3a + 3b; the dataset layer that drives the same tools (see [`docs/astribot/`](../../astribot/)) |
| `tools/general_test/pipeline/run_object_detection.py` | 3a alone | episode layout or `--keyframes-dir` |
| `tools/general_test/pipeline/run_object_init_points.py` | 3b alone | episode layout or `--keyframes-dir`; reads the prompts + key-frames from the Step-3a detections JSON (no JSON-less fallback) |

## Quickstart

```bash
# Folder mode (Case 1): one sub-task's key-frame folder, full Step 3
python tools/general_test/pipeline/run_e2e_init_points.py \
    --keyframes-dir astri_making_coffee_v1/eps_data/sampling_points/key_frames/ep000000/subtask_00/cam_head
# or the wrapper script (extra args pass through)
bash scripts/general_test/infer_step3.sh <key-frames-dir> [extra args...]

# Dataset mode (Case 2): episodes on the dataset, extraction included;
# prompts come from <data-root>/meta/subtasks.csv (required).
# --camera-idxes selects the cameras of the whole pipeline: Step 1
# extracts their key-frames, and 3a/3b run on each of them — outputs are
# per camera (detections/ep*/<camera>.json, init_points/.../<camera>/)
python tools/astribot/run_step3_init_points.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 --episode-idxes 0 \
    --camera-idxes 0 4 5

# Re-run only 3b with tuned params, reusing the saved key-frames + detections
# (--skip-3a requires the detections JSON on disk — missing -> error, in
# both drivers: no silent text-only fallback)
python tools/general_test/pipeline/run_e2e_init_points.py \
    --keyframes-dir .../subtask_00/cam_head --skip-3a --top-k 64

# Dataset mode: --skip-3a implies --skip-extract — the reused detections
# were made from the key-frames on disk, so Step 1 is skipped too. The
# reused key-frames must carry their Step-1 subtask_labels.json per
# episode and the detections JSON must exist (missing -> error, in the
# episode driver: no ordinal fallback)
python tools/astribot/run_step3_init_points.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 --episode-idxes 0 \
    --skip-3a --top-k 64

# A/B the RoMAv2 candidate pool (3b-only re-run): --sampling-mode mask
# (default, both drivers) samples inside the object masks; --sampling-mode
# uniform samples over the whole enlarged crops and the in-mask top-k
# filter decides. Same crops, same filter — run both modes into separate
# output roots to compare (--detections-dir keeps the 3b-only re-run
# reading the original detections; the default detections/init-points
# roots are <data-root>/eps_data/sampling_points/{detections,init_points})
python tools/general_test/pipeline/run_object_init_points.py \
    --data-root /data/astri_making_coffee_v1 --episode-idxes 0 \
    --sampling-mode uniform \
    -o /data/astri_making_coffee_v1/eps_data_uniform \
    --detections-dir /data/astri_making_coffee_v1/eps_data/sampling_points/detections
```

## Expected output

Step 3a — one JSON per **episode and camera** (every --camera-keys entry
with key-frames on disk is detected; a multi-camera run never overwrites
itself), the hard-filtered RexOmni predictions (Step 3b reads it). Every
sub-task entry records its `prompts` next to its `detections` (the
detections are keyed by prompt text). Per category per frame two hard
filters run after inference (see `_refine_detections` in the tool): boxes
that duplicate one instance — same image half, centers within 20% of the
image width — merge into their union (one hand occasionally fires twice),
and when the prompt names a side (`left`/`right` robot arm) only that
side's box is kept (the model often returns both arms for a side prompt).
The JSON therefore holds at most one box per side-named category per
frame; the tool's per-key-frame log notes what fired (`merge 2->1`,
`left-keep 2->1`):

```
<out-dir>/detections/ep{episode_idx:06d}/<camera>.json
```

Step 3b — per sub-task, per prompt, per camera:

```
<out-dir>/init_points/ep{ep:06d}/subtask_{k:02d}/<camera>/<prompt_slug>/
    init_points.npz    keypoints (K, N, 2) px  + frame_indices, masks, boxes, scores
    masks_rle.json     SAM3 masks as COCO-style RLE (portable reuse)
    init_points.json   metadata (episode, segment, camera_key, keyframes,
                       num_keypoints, top_k, bbox_scale, sampling_mode,
                       empty_reason on failure)
    viz.png            key-frames with masks, boxes and the tracks
```

`<camera>` is the camera subdir name (e.g. `cam_head`) — the same nesting
as the key-frames and Step 2's `depth_<camera>` folders, so a run over
several cameras writes one subtree per camera instead of overwriting.
Folder mode (`--keyframes-dir`, one camera by construction) writes the
flat `<prompt_slug>/` under `subtask_00/`.

`<prompt_slug>` is the text prompt slugified (e.g. `brown_coffee_cup`).
Failures are recorded in `init_points.json` (`empty_reason`) — the schema is
uniform, so downstream consumers always find the same files.

## Verification checklist

- [ ] Exit code 0.
- [ ] Step 3a wrote one `<out-dir>/detections/ep*/<camera>.json` per
      (episode, camera) with one entry per key-frame; the frame indexes
      inside match the `frame_<idx>` stems; in dataset mode each sub-task
      entry carries its `subtask_index` (its canonical label from
      `subtask_labels.json`) and the `prompts` of that label's row in
      `meta/subtasks.csv` together with the aligned `prompt_roles`
      ([object, manipulator] — e.g. segment `subtask_01` labelled `2`
      gets row 2's prompts, not row 1's); folder mode records no roles.
- [ ] Step 3b wrote one `init_points/` subtree per camera (under
      `ep*/subtask_XX/<camera>/`) with all four files per prompt
      (`init_points.npz`, `masks_rle.json`, `init_points.json`,
      `viz.png`).
- [ ] `init_points.npz` keypoints are `(K, N, 2)` — K ≤ `--top-k` points
      visible in all N key-frames (N = number of matched key-frames
      (dataset mode: 2 for an object prompt of a canonical 4-frame
      sub-task — the close/open pair — all key-frames otherwise)).
- [ ] `viz.png` shows the tracks: for each key-frame, the object mask, the
      bbox, and the K keypoints — keypoints lie **inside the object masks**
      (default `--sampling-mode mask` constrains the RoMAv2 pool to the
      masks; `--sampling-mode uniform` samples the whole crops and the
      in-mask top-k filter keeps the tracks — the K points stay in-mask
      either way).
- [ ] Re-run with `--skip-3a` reuses the per-camera detections JSONs (no
      new detection pass — 3a is skipped) and implies `--skip-extract`
      (no re-extraction either — 3b reuses the key-frames on disk).
- [ ] `--skip-3a` without a selected camera's detections JSON on disk
      exits with an error naming the missing (episode, camera) pairs — no
      silent text-only fallback.
- [ ] Episode driver: `--skip-extract` (passed or implied by `--skip-3a`)
      without the key-frames or their `subtask_labels.json` on disk exits
      with an error asking to drop the skip flag(s) — Step 3a never falls
      back to segment-ordinal prompts.

## Module pointers

- [`rexomni.md`](../module/rexomni.md) — 3a: RexOmni detection, the
  `.venv-rexomni` environment setup (`scripts/general_test/setup_rexomni_env.sh`)
- [`sam3.md`](../module/sam3.md) — 3b: SAM3 promptable segmentation
  (box + text prompts)
- [`romav2.md`](../module/romav2.md) — 3b: RoMAv2 cross-key-frame keypoint
  matching (`--strategy reference|cycle`, `--num-corresp`, mask-constrained
  sampling via `masks=[mask_A, mask_B]`)
