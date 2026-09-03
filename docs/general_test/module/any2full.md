# RGB-D Depth Densification — Any2Full (`tools/general_test/module/infer_any2full.py`)

Single-frame RGB-D densification on an Astribot frame: uses the RGB-D
sensor's sparse metric depth as a *prompt* to ground a Depth Anything
prediction, recovering the metric scale with a deterministic affine fit, and
exports a coloured point cloud. Runs the `.pt2` runtime
(`Any2Full_PT2`: preprocess → infer → postprocess). Inputs are resized to
the exported fixed 480×640, so any RGB/depth size is accepted; output depth
and the point cloud are at the exported resolution.

```bash
python tools/general_test/module/infer_any2full.py \
    --pt2 weights/any2full/Any2Full_vitl_bf16.pt2 \
    --frame_idx 0 \
    --out_dir ./output/a2f
```

| Argument | Default | Description |
|---|---|---|
| `--pt2` | `weights/any2full/Any2Full_vitl_bf16.pt2` | `torch.export` checkpoint |
| `--camera_name` | `head_rgbd` | Astribot camera (`head_rgbd`, `torso_rgbd`, wrists); frames loaded via `utils.astribot_dataloader.load_rgbd` (metric depth in metres + calibrated intrinsics/extrinsics) |
| `--frame_idx` | `0` | Frame index to load |
| `--out_dir` | `./output/a2f` | Output directory |
| `--denoise` / `--denoise_threshold` / `--denoise_kernel_size` / `--denoise_min_valid` | off / `2.0` / auto / `5` | Denoise the sparse depth before inference (drops isolated/anomalous points) |
| `--init_scaling` | `True` | Enable the deterministic affine scale recovery against the sparse anchors; off = raw graph depth |
| `--max_depth` / `--min_depth` | `10` / `0` | Clamp the final output (metres) |

**Output** — `frame_<idx>.glb` (coloured point cloud back-projected with the
camera intrinsics), `frame_<idx>.npy` (metric depth), `<stem>.png`
(colour-mapped depth).

## Usage in the pipeline

Any2Full is the **`a2f` streaming backend**: `run_depth_stream.py --backend a2f`
feeds the raw sensor depth (`.lz4`, log-encoded uint8 — decoded to float
metres over [0.001, 2.001] m by `load_depth_lz4`) of each RGB camera as the
prompt and outputs densified metric depth + pose (see
[`streaming.md`](streaming.md)).
