# Text-Prompt Segmentation — SAM3 (`tools/general_test/module/infer_sam3.py`)

Runs the exported SAM3 image model (`sam3_image_exported_bf16.pt2`, fixed
1008 resolution) as a plain CLI on a **folder of images** (or a **single
image file**) with a list of text prompts — one `predict()` call per prompt,
and **one overlay PNG per input image** with all detected instances (masks +
boxes + scores) drawn together.

```bash
# Every image of the folder × every prompt -> one overlay per image
python tools/general_test/module/infer_sam3.py \
    -i assets/astribot_test_imgs/head_stereo_left \
    --prompts "brown cup" "coffee machine" \
    --out_dir ./output/sam3

# Single image file
python tools/general_test/module/infer_sam3.py \
    -i assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg \
    --prompts "brown cup" --out_dir ./output/sam3

# Single frame of a folder only (frame-idx is a 0-based index into the
# sorted file list, not a frame number)
python tools/general_test/module/infer_sam3.py \
    -i assets/astribot_test_imgs/head_stereo_left --frame-idx 2 \
    --prompts "brown cup" --out_dir ./output/sam3
```

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | **required** | A single image file, or a folder of images (`.jpg/.jpeg/.png/.bmp/.webp`, sorted by filename) |
| `--frame-idx` | `None` | 0-based index into the sorted image list — process only that frame; default: all images (a single-file input is a 1-element list, so only `0` is valid) |
| `--prompts` | **required** | One or more text prompts (`nargs='+'`); each runs its own `predict()` call |
| `--conf` | `0.5` | Detection confidence threshold passed to `predict()` |
| `--pt2` | `weights/sam3/sam3_image_exported_bf16.pt2` | Exported graph checkpoint |
| `--device` | `cuda` | Device (must be CUDA) |
| `--out_dir` | `./output/sam3` | Output folder |

**Output** — flat under `--out_dir`, one PNG per input image
(`<stem>.png`, full input resolution, headless `Agg` backend): per-instance
masks overlaid with the `COLORS` palette and box + score labels
(`[p] id=i, score` — `p` is the prompt index, so the same object id under
different prompts gets a different colour). No file is written when no
prompt detects an object on that image; the per-image console log reports
each prompt's object count and per-instance score + xyxy box.

Key points:

- `predict()` returns a state dict with `masks` (binarized, original
  resolution), `boxes` (XYXY px), `scores`.
- The graph is exported at a fixed 1008 resolution with a fixed number of
  box-prompt slots; text-only prompting needs no prompt packing.
- Matplotlib is used only to compose the overlay PNG (no interactive
  `plt.show()`); running images stay on GPU, viz tensors are moved to CPU.

## Usage in the pipeline

SAM3 provides the **object prompt** for TAPIP3D 3D tracking:
`infer_tapip3d.py --bbox ... --text_prompt ...` segments the tracked object's
box on the first frame and samples the query points inside the mask (see
[`tapip3d.md`](tapip3d.md)).
