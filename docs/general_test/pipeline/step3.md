# Pipeline Test · Step 3 — Sampling Keypoints

End-to-end test of **Step 3 (Sampling Keypoints)** of the repo README: for
each interacted object, detect the object on the sub-task's key-frames,
segment its masks, match keypoints across the key-frames, and keep the
top-k keypoints inside the masks.

Step 3 runs on the **key-frames that Step 1 saved to disk**
(`tools/astribot/extract_frames.py --mode key_frames`) — a handful of frames
per sub-task, so they are persisted instead of decoding the episode videos
online (unlike Step 2, which streams every frame). The on-disk layout is:

```
<keyframes_root>/ep{ep:06d}/subtask_{k:02d}/<camera>/frame_<idx:06d>.jpg
```

## What the step does

```
text prompt per interacted object (from the subtask description)
  → RexOmni     — detect the object in each key-frame            (Step 3a)
  → SAM3        — segment the object masks (bbox + text prompt)  (Step 3b)
  → RoMAv2      — match keypoints across key-frames on enlarged
                  bbox crops (mask cropped with the same box, so
                  points are sampled inside the object only)
  → top-k keypoints inside the object masks
```

Two sub-steps, two environments — they **cannot share a process**:

| Sub-step | Tool | What it runs | Environment |
|---|---|---|---|
| 3a | `tools/general_test/run_subtask_detections.py` | RexOmni open-vocabulary detection → one JSON per episode | `.venv-rexomni` (Python 3.10 / torch 2.7, see [`rexomni.md`](../module/rexomni.md)) |
| 3b | `tools/general_test/run_subtask_init_points.py` | SAM3 masks + RoMAv2 keypoints → `init_points/` per prompt | main env (`.venv`, `.pt2` runtimes) |

The high-level drivers launch the two tools **sequentially as
subprocesses**, so you run them from the **main** environment:

```
run_folder_step3.py / run_subtask_step3.py  (drivers)
    ├── 3a: run_subtask_detections.py  →  <out>/detections/ep{idx:06d}.json
    └── 3b: run_subtask_init_points.py →  <out>/init_points/ep{idx:06d}/subtask_XX/<prompt_slug>/
```

## Entry points

| Entry point | Scope | What it runs |
|---|---|---|
| `tools/general_test/run_folder_step3.py` | **one key-frame folder** (one sub-task of one camera, dataset-independent) | 3a + 3b; the folder is sub-task 00 of a synthetic episode labelled `--episode-idx` (default 0), the folder name is the camera key |
| `scripts/general_test/infer_step3.sh` | wrapper of `run_folder_step3.py` | `python tools/general_test/run_folder_step3.py --keyframes-dir "$1" "$@"` |
| `tools/astribot/run_subtask_step3.py` | **episodes on the dataset** | Step 1 (extract the sub-task key-frames) + 3a + 3b; the dataset layer that drives the same tools (see [`docs/astribot/`](../../astribot/)) |
| `tools/general_test/run_subtask_detections.py` | 3a alone | episode layout or `--keyframes-dir` |
| `tools/general_test/run_subtask_init_points.py` | 3b alone | episode layout or `--keyframes-dir`; `--no-rexomni` for SAM3 text-only prompts (no 3a JSON) |

## Quickstart

```bash
# Folder mode (Case 1): one sub-task's key-frame folder, full Step 3
python tools/general_test/run_folder_step3.py \
    --keyframes-dir astri_making_coffee_v1/eps_data/key_frames/ep000000/subtask_00/cam_head
# or the wrapper script (extra args pass through)
bash scripts/general_test/infer_step3.sh <key-frames-dir> [extra args...]

# Dataset mode (Case 2): episodes on the dataset, extraction included
python tools/astribot/run_subtask_step3.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee --episode-idxes 0

# Re-run only 3b with tuned params, reusing the saved key-frames + detections
python tools/general_test/run_folder_step3.py \
    --keyframes-dir .../subtask_00/cam_head --skip-3a --top-k 64

# Skip RexOmni entirely: 3b with SAM3 text-only prompts
python tools/general_test/run_folder_step3.py \
    --keyframes-dir .../subtask_00/cam_head --no-rexomni
```

## Expected output

Step 3a — one JSON per episode, the raw RexOmni predictions (Step 3b reads it):

```
<out-dir>/detections/ep{episode_idx:06d}.json
```

Step 3b — per sub-task, per prompt:

```
<out-dir>/init_points/ep{ep:06d}/subtask_{k:02d}/<prompt_slug>/
    init_points.npz    keypoints (K, N, 2) px  + frame_indices, masks, boxes, scores
    masks_rle.json     SAM3 masks as COCO-style RLE (portable reuse)
    init_points.json   metadata (episode, segment, keyframes, num_keypoints,
                       top_k, bbox_scale, empty_reason on failure)
    viz.png            key-frames with masks, boxes and the tracks
```

`<prompt_slug>` is the text prompt slugified (e.g. `brown_coffee_cup`).
Failures are recorded in `init_points.json` (`empty_reason`) — the schema is
uniform, so downstream consumers always find the same files.

## Verification checklist

- [ ] Exit code 0.
- [ ] Step 3a wrote `<out-dir>/detections/ep*.json` with one entry per
      key-frame; the frame indexes inside match the `frame_<idx>` stems.
- [ ] Step 3b wrote one `init_points/` folder per prompt with all four files
      (`init_points.npz`, `masks_rle.json`, `init_points.json`, `viz.png`).
- [ ] `init_points.npz` keypoints are `(K, N, 2)` — K ≤ `--top-k` points
      visible in all N key-frames (N = number of key-frames).
- [ ] `viz.png` shows the tracks: for each key-frame, the object mask, the
      bbox, and the K keypoints — keypoints lie **inside the object masks**
      (mask-constrained RoMAv2 sampling).
- [ ] Re-run with `--skip-3a` reuses the detections JSON (no new detection
      pass — 3a is skipped).
- [ ] With `--no-rexomni`, 3b runs on SAM3 text-only prompts and still
      produces the same output layout.

## Module pointers

- [`rexomni.md`](../module/rexomni.md) — 3a: RexOmni detection, the
  `.venv-rexomni` environment setup (`scripts/general_test/setup_rexomni_env.sh`)
- [`sam3.md`](../module/sam3.md) — 3b: SAM3 promptable segmentation
  (box + text prompts)
- [`romav2.md`](../module/romav2.md) — 3b: RoMAv2 cross-key-frame keypoint
  matching (`--strategy reference|cycle`, `--num-corresp`, mask-constrained
  sampling via `masks=[mask_A, mask_B]`)
