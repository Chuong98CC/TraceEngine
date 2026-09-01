# Pipeline Test · Step 2 — Camera Pose + Depth

End-to-end test of **Step 2 (Camera Pose and Depth)** of the repo README:
stream a folder of frames through a depth+pose backend, in sliding chunks
aligned into one world frame, and verify the per-frame `depth + pose` output
contract.

This is the **folder-mode** (Case 1) test: input is a folder of extracted
images, dataset-independent. The dataset-mode variant that streams online
from a LeRobotDataset without writing frames to disk is
`tools/astribot/run_subtask_stream.py` (see
[`docs/astribot/astribot_subtask_depth_stream.md`](../../astribot/astribot_subtask_depth_stream.md)).
The model-level details live in the module docs — this page is about the
pipeline running end-to-end.

## What the step does

```
split the frame sequence into overlapped chunks (≤ num_views)
        → depth+pose model per chunk (backend by camera setup)
        → SIM3 chunk alignment → global camera poses + depth per frame
```

Camera setup → backend (full details in [`streaming.md`](../module/streaming.md)):

| Camera setup | Backend | Notes |
|---|---|---|
| Stereo pair | `vggt_omega` / `da3` | two input folders |
| RGB-D | `a2f` | RGB folders + parallel raw-depth folders; Any2Full densifies the sparse sensor depth, DA3 aligns to it (`--no-depth-enhance` to skip Any2Full) |
| Arbitrary RGB | `da3` / `vggt_omega` | mono input |

Optional: WAFT motion masks (`infer_waft.py`) zero the confidence of moving
pixels **during chunk alignment only** — depth outputs are unaffected
(see [`waft.md`](../module/waft.md)).

## Entry points

| Entry point | What it runs |
|---|---|
| `tools/general_test/pipeline/run_depth_stream.py` | The Step-2 pipeline itself (all backends, `--video` to render afterwards) |
| `scripts/general_test/infer_stream_stereo.sh` | Stereo streaming on the sample extraction (uncomment the `run_depth_stream.py` block), then renders the trajectory video |
| `scripts/general_test/infer_stream_rgbd.sh` | RGB-D streaming (`a2f`) on the sample extraction, then renders the trajectory video |
| `scripts/general_test/visualize_stream.sh` | Trajectory video from a streaming result (no re-inference) |

## Quickstart

```bash
# Stereo (VGGT-Omega or DA3) — frame folders from Step 1
python tools/general_test/pipeline/run_depth_stream.py \
    --backend vggt_omega \                 # or "da3"
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --start-frame 210 --max-frames 160 --interval 4 \
    --output-dir output/stream_stereo_vggt_omega

# RGB-D (Any2Full): RGB folder + raw-depth folder pair
python tools/general_test/pipeline/run_depth_stream.py \
    --backend a2f \
    --input-dirs <rgb_frames_dir> \
    --depth-dirs <raw_depth_dir> \
    --start-frame 0 --max-frames 160 --interval 1 \
    --output-dir output/stream_rgbd_a2f

# Render the trajectory video from the saved result (no re-inference)
python tools/general_test/pipeline/visualize_stream.py \
    --input-dirs <left_frames_dir> <right_frames_dir> \
    --result-dir output/stream_stereo_vggt_omega \
    --output output/stream_stereo_vggt_omega/trajectory.mp4 \
    --fps 30 --size 960x540
```

To get sample frame folders, extract them from a LeRobotDataset first
(`tools/astribot/extract_frames.py --mode frames`, see
[`docs/astribot/astribot_extract_frames.md`](../../astribot/astribot_extract_frames.md)).

## Expected output

Per camera: `<output-dir>/depth_<camera_name>/` with, per frame

- `frame_<idx>.lz4` — depth, raw **uint16 mm**
- `frame_<idx>.npz` — pose: `extrinsics` (3×4 world→camera), `intrinsics`
  (3×3), plus `shape` (the depth shape the lz4 buffer is reshaped to)

plus `timings.json`, and the run finishes with a timing/memory summary
(total time, per-chunk mean/min/max, chunks/s, peak GPU memory, peak CPU RSS).

For `--backend a2f` only the RGB folders emit `depth_<name>/` — the depth
folders are inputs only.

This output contract is what Step 4 (`infer_tapip3d.py`) consumes for 3D
tracing and what `visualize_rgbd.py` / `visualize_stream.py` render.

## Verification checklist

- [ ] Exit code 0; the run prints the timing/memory summary.
- [ ] One `depth_<camera_name>/` per input folder, with `frame_<idx>.lz4` +
      `frame_<idx>.npz` for every processed frame (all cameras emit the same
      frame stems — synchronized).
- [ ] `frame_<idx>.npz` contains `extrinsics` (3×4), `intrinsics` (3×3) and
      `shape`; `frame_<idx>.lz4` reshaped to `shape` is a non-empty uint16 map
      (metric depth in mm, 0 = invalid).
- [ ] Depth values look sane for the scene (e.g. valid pixels ≈ the observed
      distances, no large all-zero frames).
- [ ] Consecutive frames' `extrinsics` are continuous (no per-chunk jumps) —
      the SIM3 alignment worked. A trajectory video makes this visible:
- [ ] `visualize_stream.py` renders a trajectory mp4 without errors, showing
      a growing camera path with consistent point clouds.

## Module pointers

- [`streaming.md`](../module/streaming.md) — `run_depth_stream.py` full CLI (backends,
  masks, alignment config, `--video`) + output contract details
- [`any2full.md`](../module/any2full.md) — the `a2f` RGB-D depth-densification
  component
- [`waft.md`](../module/waft.md) — WAFT motion masks (optional `--mask-dirs`)
- [`visualize_depth_stream.md`](../module/visualize_depth_stream.md) — the trajectory
  renderer (viewpoints, tone-curve inversion, tuning flags)
- [`visualize_rgbd.md`](../module/visualize_rgbd.md) — RGB-D frame renderer
- [`tapip3d.md`](../module/tapip3d.md) — Step 4 consumes this output contract
