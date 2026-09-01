# TAPIP3D 3D Point Tracking

Streaming long-video 3D point tracking with the TAPIP3D **torch.export**
programs: the encoder and the fused corr+updater iteration run as `.pt2`
graphs in sliding windows of 16 frames, orchestrated by `Tapip3DStreamPT2`.
Image size and query count are derived from the graph input shapes.

Queries are anchored at the first frame: points inside the tracked-object
bounding box are unprojected to world coordinates with the first frame's
depth + pose, and a full-frame support grid is appended. The tracks are
saved as world-space 3D traces per query.

```bash
# Grid queries (classic): 8x8 bbox grid + 32x32 support grid
python tools/general_test/module/infer_tapip3d.py \
    --image_dir \
        /data/astri_making_coffee_v1/eps_data/subtask_frames/ep000000/subtask_00/cam_head \
    --depth_dir \
        /data/astri_making_coffee_v1/experiments/rgbd_a2f/depth_cam_head \
    --bbox 1 240 100 340 \
    --grid_x 8 --grid_y 8 --support_grid_size 32 \
    --output_dir output/stream_tracks_pt2

# SAM3 mask-guided queries: sample the bbox points inside the SAM3
# segmentation of the object instead of on a regular grid
python tools/general_test/module/infer_tapip3d.py \
    --image_dir <frames_dir> --depth_dir <geometry_dir> \
    --bbox 1 240 100 340 --text_prompt "brown coffee cup" \
    --grid_x 8 --grid_y 8 --support_grid_size 32 \
    --visualize --output_dir output/stream_tracks_sam3
```

Wrapper: `scripts/general_test/infer_tapip3d.sh` (env vars `IMG_DIR` /
`DEPTH_DIR` / `OUTPUT_DIR`; tracks the head camera with
`--bbox 1 240 100 340 --text_prompt "brown coffee cup"`).

## Query modes

| Mode | When | Bbox query points |
|---|---|---|
| `grid` | no `--bbox` (whole frame), or segmentation fallback | regular `grid_x × grid_y` grid inside the box |
| `mask` | `--bbox` given and SAM3 segments the box | `grid_x × grid_y` points sampled **uniformly at random inside the SAM3 mask** |

In `mask` mode the object in the box is segmented on the first frame with
`--sam3_checkpoint` (default `weights/sam3/sam3_image_exported_bf16.pt2`)
using the box + `--text_prompt` (default `"visual"`) — the SAM3 image model,
see [`sam3.md`](sam3.md). Sampling is restricted
to mask pixels with **valid depth** (`> 0`), so every sampled point survives
the unprojection and the query count stays exact. If SAM3 produces no mask,
or the mask has fewer valid-depth pixels than `grid_x × grid_y`, the script
falls back to `grid` mode with a warning. `--seed` (default 0) makes the
sampling reproducible. The mask is saved as `sam3_mask.npy` plus a
visualization overlay `sam3_mask.png` (first frame with box, mask and the
sampled query points).

> **Exact-N contract** — the shipped iteration program
> (`weights/tapip3d/tapip3d_iteration_1088_bf16.pt2`) has a **fixed query
> count (1088)**: 8×8 bbox points (64) + 32×32 support grid (1024). The
> script exits with a clear error if the count doesn't match — e.g. depth
> holes dropping `grid`-mode bbox points, different
> `--grid_x/--grid_y/--support_grid_size` values, or a mask too small to
> sample `grid_x × grid_y` points from.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--image_dir` | **required** | Folder of frames to track in |
| `--depth_dir` | **required** | Geometry folder: per-frame `frame_<idx>.lz4` (depth, raw uint16 mm) + `frame_<idx>.npz` with `extrinsics` (3×4 or 4×4, world→camera), `intrinsics` (3×3), `shape` |
| `--output_dir`, `-o` | `output/stream_tracks_pt2` | Output directory |
| `--encoder` | `weights/tapip3d/tapip3d_encoder_480x640_bf16.pt2` | Encoder `.pt2` (image size asserted against `--image_size`) |
| `--iteration` | `weights/tapip3d/tapip3d_iteration_1088_bf16.pt2` | Fused corr+updater `.pt2` (query count auto-detected from the graph) |
| `--image_size` | `480 640` | Inference resolution (H W), must match the encoder graph |
| `--start_frame` / `--max_frames` / `--interval` | `0` / all / `1` | Sequence subsampling |
| `--bbox` | full frame | `x0 y0 x1 y1` of the tracked object; enables SAM3 `mask` mode |
| `--grid_x` / `--grid_y` | `8` / `8` | Bbox query points (grid size in `grid` mode, sampled count in `mask` mode) |
| `--support_grid_size` | `32` | Full-frame support grid (added to the bbox queries) |
| `--sam3_checkpoint` | `weights/sam3/sam3_image_exported_bf16.pt2` | SAM3 image `.pt2`, segments the box on the first frame when `--bbox` is given |
| `--text_prompt` | `visual` | SAM3 text prompt for the box segmentation |
| `--seed` | `0` | RNG seed for the in-mask query sampling |
| `--num_iters` | `6` | Fused corr+updater iterations inside each window |
| `--vis_threshold` | `0.5` | Sigmoid visibility threshold for `visibs` |
| `--visualize` / `--video_fps` | off / `10` | Render the tracks as a video after inference |

## Output

| File | Content |
|---|---|
| `coords.npy` | (T, Q, 3) world-space 3D traces |
| `visibs.npy` | (T, Q) per-query visibility (thresholded at `--vis_threshold`) |
| `metadata.json` | paths, grid, bbox, `query_source` (`mask`/`grid`), `text_prompt`, `sam3_mask_files`, `seed`, frame indices |
| `sam3_mask.npy` / `sam3_mask.png` | `mask` mode only: segmentation mask + first-frame overlay (box, mask, sampled query points) |
| `tracks.mp4` | with `--visualize`: the tracks over the video |

## Notes

- Bbox hints (Astribot head camera, 640×480): head `1 240 100 340`, torso
  `70 357 133 396`, left stereo `200 265 260 300`.
- The script runs the TAPIP3D path with TF32 **off** (validated numerics).
  The SAM3 call temporarily re-enables TF32 and the flash/mem-efficient SDPA
  kernels (importing `flow_models.tapip3d` disables them globally), then
  restores both — keep this in mind if reusing `Sam3Image` in other scripts
  that import tapip3d.
- Geometry is a `run_depth_stream.py`-style output folder (see
  [`streaming.md`](streaming.md) for the streaming pipeline that produces
  it).
