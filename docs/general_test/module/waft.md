# Optical Flow / Motion Masks — WAFT (`tools/general_test/infer_waft.py`)

Dense optical flow on a **video file or a folder of extracted frames**, with
ONNX or TensorRT backends. Folder input is auto-detected when `--input` is a
directory: frames are scanned as `frame_{idx}.jpg` / `frame_{idx}.png`,
sorted by index, and per-frame outputs are named after the anchor frame's
source index. Backend is inferred from the checkpoint file extension
(`.onnx` → ONNX Runtime, `.engine` → TensorRT); pass `--backend` when the
path has no extension.

```bash
# Folder of extracted frames, TensorRT, motion masks (pixels > 2 px moving)
python tools/general_test/infer_waft.py --input <frames_dir> --backend trt \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine \
    --start 210 --stride 4 -thr 2 -o mask \
    --output-dir demo_data/astribot_stereo_lrb/motion_mask/

# Video file, ONNX, flow colour video + raw .flo files
python tools/general_test/infer_waft.py --input video.mp4 \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480.onnx \
    --output-mode all
```

| Argument | Default | Description |
|---|---|---|
| `--input` | **required** | Video file or folder of `frame_{idx}.jpg/.png` |
| `--checkpoint` | `weights/waftv2/waftv2_dinov3_i5_640x480` | Model checkpoint; backend inferred from `.onnx` / `.engine` extension |
| `--backend` | auto | `onnx` or `trt`; required only when the checkpoint has no extension |
| `--output-dir` | `./output` | Root output directory (per-input subfolder: `<output-dir>/<video_name>/`) |
| `--output-mode`, `-o` | `flow` | `flow` (colour video), `raw` (`.flo` files), `overlay` (frame+flow overlay), `mask` (motion-mask frames), `flow-mask` (motion-mask video), `all` |
| `--start` | `0` | Frame **index** in folder mode, frames to skip in video mode |
| `--stride` | `1` | Gap between paired frames (≥ 1) |
| `--max-frames` | until EOF | Max number of frame **pairs** to process |
| `--motion-threshold`, `-thr` | — | Pixel-displacement threshold for the binary moving-pixel mask (required for `mask` / `flow-mask`) |
| `--device` | `cuda` | ONNX Runtime device (`cuda` / `cpu`); ignored for TRT |
| `--no-bgr-input` | off | Images are already RGB (skip internal BGR→RGB conversion) |

**Output** — under `<output-dir>/<video_name>/`: `flow/flow.mp4` (colour
wheel), `raw/frame_*.flo` (Middlebury format), `overlay/overlay.mp4`,
`mask/mask_<idx>.jpg` (255 = moving, >127 is moving in the binary masks
consumed by the streaming step) and `mask/mask.mp4`.

Wrapper script: `scripts/general_test/infer_waft.sh` (TRT engine, motion
masks on a demo frame folder).

## Notes

- Inputs use the same letterbox pipeline as DA3 (no ImageNet normalization).
  Flow is rescaled and cropped back to the original resolution on output.
- **Flow format**: `[H, W, 2]` float32 array where `[:,:,0]` = horizontal
  displacement (dx) and `[:,:,1]` = vertical displacement (dy). Raw output is
  saved in the standard `.flo` (Middlebury) format.
- **Motion masks in streaming**: the binary masks produced here feed
  `run_stream.py --mask-dirs` — moving pixels' confidence is zeroed during
  chunk alignment only (see [`streaming.md`](streaming.md)).
