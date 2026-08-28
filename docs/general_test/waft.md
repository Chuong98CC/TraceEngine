# WAFT Optical Flow

Dense optical flow on video frame pairs (ONNX or TensorRT). The model estimates
per-pixel displacement between consecutive frames and can generate colour-wheel
visualisations, raw `.flo` files, frame overlays, and binary motion masks.

Backend is auto-detected from the checkpoint file extension (`.onnx` → ONNX
Runtime, `.engine` → TensorRT). If the path has no recognised extension, pass
`--backend` explicitly.

```bash
# TensorRT with the default engine
python tools/infer_waft.py --input video.mp4 --backend trt

# ONNX with an explicit checkpoint
python tools/infer_waft.py --input video.mp4 \
    --checkpoint weights/waftv2/waftv2_dinov3_i5_640x480.onnx

# Generate flow colour video + raw .flo files
python tools/infer_waft.py --input video.mp4 --backend trt --output-mode flow

# Skip the first 100 frames, process every 5th pair
python tools/infer_waft.py --input video.mp4 --backend trt --start 100 --stride 5

# Motion mask for pixels moving more than 2 px
python tools/infer_waft.py --input video.mp4 --backend trt \
    --motion-threshold 2 --output-mode all
```

| Argument | Default | Description |
|---|---|---|
| `--input` | **required** | Path to input video file |
| `--checkpoint` | `weights/waftv2/waftv2_dinov3_i5_640x480` | Model checkpoint (`.onnx` / `.engine`); backend inferred from extension |
| `--backend` | auto | `onnx` or `trt`; required only when `--checkpoint` has no extension |
| `--output-dir` | `./output` | Root output directory |
| `--output-mode`, `-o` | `flow` | `flow` (colour video), `raw` (`.flo` files), `overlay` (frame+flow overlay), `mask` (motion-mask frames), `flow-mask` (motion-mask video), or `all` |
| `--start` | `0` | Skip first N frames |
| `--stride` | `1` | Gap between paired frames (≥ 1) |
| `--max-frames` | until EOF | Max number of frame **pairs** to process |
| `--motion-threshold`, `-thr` | — | Pixel-displacement threshold for a binary moving-pixel mask |
| `--device` | `cuda` | ONNX Runtime device (`cuda` or `cpu`); ignored for TRT |
| `--no-bgr-input` | off | Images are already RGB (skip internal BGR→RGB conversion) |

## Notes

- Inputs use the same letterbox pipeline as DA3 (no ImageNet normalization). Flow
  is rescaled and cropped back to the original resolution on output.
- **Flow format**: `[H, W, 2]` float32 array where `[:,:,0]` = horizontal
  displacement (dx) and `[:,:,1]` = vertical displacement (dy). Raw output is
  saved in the standard `.flo` (Middlebury) format.
- **Output modes**: `flow` produces a colour-wheel video; `raw` writes
  per-frame `.flo` files; `overlay` composites the flow colour map over the
  source frame; `mask` / `flow-mask` generate binary motion masks (requires
  `--motion-threshold`).
