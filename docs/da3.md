# Depth Anything 3 (DA3)

Monocular and multi-view depth estimation with three modules.

Run a module (`metric` / `anyview` / `nested`) on a backend (`onnx` / `trt`).
Results are saved as a `result.npz` per frame, cropped to each view's tile.

```bash
# Nested pipeline, ONNX (default); the model predicts its own camera poses
python tools/infer_da3.py --module nested --camera-set set1 --frame 0

# Any-view branch, TensorRT
python tools/infer_da3.py --backend trt --module anyview --frame 0

# Metric model over all frames
python tools/infer_da3.py --module metric --all-frames
```

| Argument | Default | Description |
|---|---|---|
| `--backend` | `onnx` | `onnx` or `trt` |
| `--module` | `nested` | `metric`, `anyview`, or `nested` |
| `--camera-set` | auto | `set0`/`set1`/`set2`; auto-matched to the model's view count if omitted (metric → `set1`) |
| `--frame` | — | Single frame index (0-based) |
| `--all-frames` | off | Process all frames common to the selected cameras |
| `--use-extrinsics` | off | Any-view/nested: select the `-with-camera-pose` model and feed camera pose priors. Omit → plain model, poses predicted |
| `--keep-predicted-pose` | off | Nested + `--use-extrinsics`: keep the model's predicted poses (rigidly aligned into the input frame) instead of replacing them with the input poses and rescaling depth |
| `--no-align-input-ext-scale` | alignment on | Nested only: disable the Umeyama alignment to input camera poses |
| `--anyview-model` / `--metric-model` | backend defaults | Override model paths |
| `--export-dir` | `output/da3` | Output directory |
| `--visualize` | off | Save colour-coded depth maps + (anyview/nested) `scene.glb` |
| `--show-cameras` | off | Draw camera frustums in the GLB (with `--visualize`) |

## Notes

- **Metric depth** in metres is a caller-side step:
  `metric_depth = focal * depth / 300`, `focal = (fx + fy) / 2`. The model returns
  raw network depth + sky.
- Inputs are letterbox-preprocessed (aspect-preserving resize + centre-pad) with
  ImageNet normalization; intrinsics are adjusted for the scale + pad and
  un-padded on output.

(Obsolete) Whether camera pose is actually fed is read from the loaded model's inputs, so
`--use-extrinsics` mainly picks the `-with-camera-pose` checkpoint. This gives the worst result. Only keep for legacy code.
