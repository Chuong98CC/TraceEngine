# Optical Flow / Motion Masks — WAFT (`tools/general_test/module/infer_waft.py`)

Dense optical flow on a **video file or a folder of extracted frames**, with
the WAFTv2 `torch.export` (`.pt2`, bf16) backend. Folder input is
auto-detected when `--input` is a directory: frames are scanned as
`frame_{idx}.jpg` / `frame_{idx}.png`, sorted by index, and per-frame
outputs are named after the anchor frame's source index. A `.pt2`
checkpoint path is used as-is; extension-less paths get `.pt2` appended
(legacy `.onnx` / `.engine` checkpoints are rejected).

```bash
# Folder of extracted frames, motion masks (pixels > 2 px moving)
python tools/general_test/module/infer_waft.py --input <frames_dir> \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480_bf16.pt2 \
    --start 210 --stride 4 -thr 2 -o mask \
    --output-dir demo_data/astribot_stereo_lrb/motion_mask/

# Video file, flow colour video + raw .flo files (default checkpoint)
python tools/general_test/module/infer_waft.py --input video.mp4 \
    --output-mode all
```

| Argument | Default | Description |
|---|---|---|
| `--input` | **required** | Video file or folder of `frame_{idx}.jpg/.png` |
| `--checkpoint` | `weights/waftv2/waftv2_dinov3_i5_640x480` | WAFTv2 `.pt2` artifact; a `.pt2` path is used as-is, otherwise `.pt2` is appended |
| `--output-dir` | `./output` | Root output directory (per-input subfolder: `<output-dir>/<video_name>/`) |
| `--output-mode`, `-o` | `flow` | `flow` (colour video), `raw` (`.flo` files), `overlay` (frame+flow overlay), `mask` (motion-mask frames), `flow-mask` (motion-mask video), `all` |
| `--start` | `0` | Frame **index** in folder mode, frames to skip in video mode |
| `--stride` | `1` | Gap between paired frames (≥ 1) |
| `--max-frames` | until EOF | Max number of frame **pairs** to process |
| `--motion-threshold`, `-thr` | — | Pixel-displacement threshold for the binary moving-pixel mask (required for `mask` / `flow-mask`) |
| `--device` | `cuda` | Device to run the `.pt2` artifact on (`cuda` / `cpu`) |

Inputs follow the repo-wide image contract: RGB pixel space via
`utils.image_io` (paths, PIL images, RGB HWC numpy arrays or CHW tensors are
all accepted; cv2 frames are converted to RGB at the call site).

**Output** — under `<output-dir>/<video_name>/`: `flow/flow.mp4` (colour
wheel), `raw/frame_*.flo` (Middlebury format), `overlay/overlay.mp4`,
`mask/mask_<idx>.jpg` (255 = moving, >127 is moving in the binary masks
consumed by the streaming step) and `mask/mask.mp4`.

Wrapper script: `scripts/general_test/infer_waft.sh` (`.pt2` artifact,
motion masks on a demo frame folder).

## Notes

- Inputs use the shared trunc2 letterbox from `utils.image_io` (same geometry
  as DA3; no ImageNet normalization — the exported graph normalizes
  internally, so the bf16 feed is in [0, 255]). Flow is rescaled and cropped
  back to the original resolution on output.
- **Flow format**: `[H, W, 2]` float32 array where `[:,:,0]` = horizontal
  displacement (dx) and `[:,:,1]` = vertical displacement (dy). Raw output is
  saved in the standard `.flo` (Middlebury) format.
- **Motion masks in streaming**: the binary masks produced here feed
  `run_depth_stream.py --mask-dirs` — moving pixels' confidence is zeroed during
  chunk alignment only (see [`streaming.md`](streaming.md)).
