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
                  --sampling-mode no_roma skips RoMAv2 entirely:
                  the simple baseline sampling top-k points inside
                  the mask of the span's first frame only
                  (manipulator: the 1st key-frame; object: the 2nd,
                  first of its close..open span), with the same
                  masks and outputs otherwise — the manipulator's
                  draw weighted toward the manipulated object (see
                  below), every other prompt uniform;
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
    --keyframes-dir astribot_making_coffee_vlva_full/eps_data/sampling_points/key_frames/ep000000/subtask_00/cam_head
# or the wrapper script (extra args pass through)
bash scripts/general_test/infer_step3.sh <key-frames-dir> [extra args...]

# Dataset mode (Case 2): episodes on the dataset, extraction included;
# prompts come from <data-root>/meta/subtasks.csv (required).
# --camera-idxes selects the cameras of the whole pipeline: Step 1
# extracts their key-frames, and 3a/3b run on each of them — outputs are
# per camera (detections/ep*/<camera>.json, init_points/.../<camera>/)
python tools/astribot/run_step3_init_points.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes 0 \
    --camera-idxes 0 4 5

# Re-run only 3b with tuned params, reusing the saved key-frames + detections
# (--skip-3a requires the detections JSON on disk — missing -> error, in
# both drivers: no silent text-only fallback). --visualize adds the rendered
# images (viz.png, and union_mask.png under --with-optical-flow-mask) — a run
# without it writes no images at all
python tools/general_test/pipeline/run_e2e_init_points.py \
    --keyframes-dir .../subtask_00/cam_head --skip-3a --object-top-k 64 \
    --visualize

# Dataset mode: --skip-3a implies --skip-extract — the reused detections
# were made from the key-frames on disk, so Step 1 is skipped too. The
# reused key-frames must carry their Step-1 subtask_labels.json per
# episode and the detections JSON must exist (missing -> error, in the
# episode driver: no ordinal fallback)
python tools/astribot/run_step3_init_points.py \
    --repo-id Kronze157/astri_making_coffee_vlva \
    --data-root /data/astri_making_coffee_v1 --episode-idxes 0 \
    --skip-3a --object-top-k 64

# A/B the RoMAv2 candidate pool (3b-only re-run): --sampling-mode mask
# (default) samples inside the object masks; --sampling-mode uniform
# samples over the whole enlarged crops and the in-mask top-k filter
# decides. Same crops, same filter — run both modes into separate output
# roots to compare (--detections-dir keeps the 3b-only re-run reading the
# original detections; the default detections/init-points roots are
# <data-root>/eps_data/sampling_points/{detections,init_points})
python tools/general_test/pipeline/run_object_init_points.py \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes 0 \
    --sampling-mode uniform \
    -o /data/astribot_making_coffee_vlva_full/eps_data_uniform \
    --detections-dir /data/astribot_making_coffee_vlva_full/eps_data/sampling_points/detections

# Simple-baseline A/B (no RoMaV2): --sampling-mode no_roma samples top-k
# points inside the mask of the span's first frame only (manipulator: 1st
# key-frame, weighted toward the manipulated object; object: 2nd, first of
# its close..open span, uniform) — same SAM3 masks, frame_indices and
# outputs, so Step 4 windows and gating are unchanged; only the matching is
# dropped. --no-manipulator-near-object samples the manipulator uniformly
# too (the pre-weighting baseline)
python tools/astribot/run_step3_init_points.py \
    --repo-id Kronze157/astribot_making_coffee_vlva_full \
    --data-root /data/astribot_making_coffee_vlva_full --episode-idxes 0 \
    --sampling-mode no_roma \
    -o /data/astribot_making_coffee_vlva_full/eps_data_no_roma
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
                       (--visualize only)
```

`<camera>` is the camera subdir name (e.g. `cam_head`) — the same nesting
as the key-frames and Step 2's `depth_<camera>` folders, so a run over
several cameras writes one subtree per camera instead of overwriting.
Folder mode (`--keyframes-dir`, one camera by construction) writes the
flat `<prompt_slug>/` under `subtask_00/`.

**Visualization is opt-in**: one `--visualize` flag per entry point turns
all of it on and nothing is rendered without it. In Step 3b it writes the
per-prompt `viz.png`, plus — for a manipulator whose mask the
`--with-optical-flow-mask` rescue widened — `union_mask.png` beside that
prompt's outputs (the row-0 key-frame tinted SAM-only / SAM ∩ motion /
motion-only, so the motion-only pixels are exactly what the flow added).
Step 3a' writes only its `motion_rle.json`: the union render above is the
way to eyeball a motion mask (there is no `flow.png`); the flags it
replaces were 3b's `--no-viz` / `--viz-motion-union` and the drivers'
`--visualize-motion`. The drivers forward their `--visualize` to 3b
alone — `--visualize` without `--with-optical-flow-mask` simply has no
union to draw.

`<prompt_slug>` is the text prompt slugified (e.g. `brown_coffee_cup`).
Failures are recorded in `init_points.json` (`empty_reason`) — the schema is
uniform, so downstream consumers always find the same files.

## The manipulator's near-object draw

With `--sampling-mode no_roma` the manipulator's top-k draw is **weighted
toward the manipulated object** — the arm points that end up on the
gripper/hand near the object matter for the Step-4 traces, the rest of the
mask does not. Its row-0 mask pixels (SAM3, ∪ the WAFT motion mask under
`--with-optical-flow-mask`) are weighted by `1/(1 + (d/R)²)` — `d` the
pixel distance to the object prompt's Step-3a box center **on the sampled
key-frame** (the largest box of that frame's detections; a frame without
one falls back to the first key-frame that has one), `R` the mask's median
distance to that center — then the weights are

1. normalized to a max of 1 (the pixel nearest the center weighs 1),
2. cut below their median (`w < median(w)` → 0), and
3. drawn from **without replacement**, proportionally to the weights.

`1/(1 + (d/R)²)` decreases with `d`, so step 2 keeps exactly the pixels
within `R` of the center — the mask's nearer half, `R` being the median —
and step 3 spreads the draw over that whole disc, at most 2× denser toward
the object. There is no distance constant to tune: the cut radius `R`
follows the mask itself and doubles as the falloff scale, which is what
keeps the points spread over the near half instead of collapsing onto the
few pixels closest to the object center (raw `1/(1+d²)`, `d` in pixels, is
a ~100:1 gradient at arm scale). The top-k count, the `frame_indices` span
and the masks saved to `masks_rle.json` are unchanged, so Step 4's windows
and per-frame gating are unaffected.

`--no-manipulator-near-object` restores the plain uniform draw of earlier
runs (bit-identical for the same episode/sub-task/camera/prompt, the seed
being unchanged). The object prompt itself and both RoMAv2 modes always
sample as before. The reference (`near_object_center`,
`near_object_center_frame`), the cut radius in pixels
(`near_object_radius_px` — the mask's median distance to the center, beyond
which nothing is ever drawn), the pool and kept pixel counts and the median
weight land in the manipulator's `init_points.json` — `near_object_weight`
is `false` with a `near_object_reason` when there was nothing to weight
against (no object-role prompt, e.g. folder mode, or no object detection on
any key-frame). The manipulator's `viz.png` (under `--visualize`) marks
the reference: an orange
crosshair at the box center and the cut-radius circle around it, i.e. the
disc the keypoints were drawn from (drawn on the sampled frame's panel even
when the box came from a later key-frame — `near_object_center_frame`).

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
      `ep*/subtask_XX/<camera>/`) with `init_points.npz`,
      `masks_rle.json` and `init_points.json` per prompt — plus `viz.png`
      when run with `--visualize` (and `union_mask.png` beside the
      manipulator's prompt under `--with-optical-flow-mask`, the two
      being the only images a Step-3 run writes).
- [ ] Without `--visualize` no `viz.png` / `union_mask.png` appears
      anywhere under the output root, and no `flow.png` does either
      (Step 3a' writes only `motion_rle.json`).
- [ ] `init_points.npz` keypoints are `(K, N, 2)` — K ≤ `--object-top-k` points
      visible in all N key-frames (N = number of matched key-frames
      (dataset mode: 2 for an object prompt of a canonical 4-frame
      sub-task — the close/open pair — all key-frames otherwise)).
- [ ] With `--visualize`, `viz.png` shows the tracks: for each key-frame,
      the object mask, the
      bbox, and the K keypoints — keypoints lie **inside the object masks**
      (default `--sampling-mode mask` constrains the RoMAv2 pool to the
      masks; `--sampling-mode uniform` samples the whole crops and the
      in-mask top-k filter keeps the tracks — the K points stay in-mask
      either way). With `--sampling-mode no_roma` the K sampled points lie
      inside the first span frame's mask and viz draws them on that
      frame's panel only (no cross-frame lines) — the manipulator's points
      inside the median-distance disc around the object prompt's box center
      (`--no-manipulator-near-object` lifts that), denser toward it; the
      center and the radius are recorded in its `init_points.json` and
      drawn on that panel.
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
