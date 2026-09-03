# Rex-Omni Open-Vocabulary Detection (`tools/general_test/module/infer_rexomni.py`)

Rex-Omni is a Qwen2.5-VL based multimodal model for **open-vocabulary object
detection** (and pointing / keypoint / OCR tasks) from text prompts. Unlike
the other models in this repo it is **not** converted to a torch.export / ONNX
checkpoint: it runs the original Hugging Face model
(`IDEA-Research/Rex-Omni`, downloaded to the HF cache on first use) inside a
**separate Python environment** `.venv-rexomni`.

## Why a separate environment

Rex-Omni's tested stack (`torch==2.7.0`, `transformers==4.51.3`,
`vllm==0.9.1`, Python 3.10) conflicts with the main project environment
(`torch==2.11.0`, `transformers 5.x`, Python 3.12). Set it up with:

```bash
bash scripts/general_test/setup_rexomni_env.sh
```

The script creates `.venv-rexomni` (overridable via `REXOMNI_ENV_DIR` /
`REXOMNI_PYTHON_VERSION`) and installs torch 2.7 (cu128), transformers 4.51.3,
qwen_vl_utils and the prebuilt FlashAttention wheel. If that wheel is
unavailable for your platform, choose a matching one from the
[flash-attention-prebuild-wheels](https://mjunya.com/flash-attention-prebuild-wheels/?package=FA2&python=3.10&torch=2.7&cuda=12.8)
page and replace the `flash-attn` URL in the script.

The environment is **not** an editable install of the project: the project
metadata requires Python 3.12, while Rex-Omni uses Python 3.10. Run the tool
by exposing the repo's `src/` via `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py
```

Do not install these packages into the main environment.

## Running the tool

`infer_rexomni.py` is a plain CLI (same contract as
[`infer_sam3.py`](sam3.md)): it runs on a **folder of images** (or a
**single image file**, or one frame of a folder selected by `--frame-idx`)
and saves **one annotated PNG per input image** — all detected boxes of all
categories drawn on the image, with per-category colours + labels.

Rex-Omni is a generative model: the categories are joined into **one
detection call per image** (`inference(task="detection", categories=...)`)
and the parsed output carries **no per-box confidence scores** — hence no
`--conf` threshold.

```bash
# Every image of the folder -> one annotated PNG per image
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
    -i assets/astribot_test_imgs/head_rgbd/color \
    --prompts "brown cup" "coffee machine" \
    --out_dir ./output/rexomni

# Single image file
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
    -i assets/astribot_test_imgs/head_rgbd/color/img_000000.jpg \
    --prompts "brown cup" --out_dir ./output/rexomni

# Single frame of a folder only (frame-idx is a 0-based index into the
# sorted file list, not a frame number)
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
    -i assets/astribot_test_imgs/head_rgbd/color --frame-idx 0 \
    --prompts "brown cup" --out_dir ./output/rexomni
```

or via the ready-to-run wrapper `scripts/general_test/module/infer_rexomni.sh`
(env interpreter + `PYTHONPATH` baked in).

| Argument | Default | Description |
|---|---|---|
| `--input`, `-i` | **required** | A single image file, or a folder of images (`.jpg/.jpeg/.png/.bmp/.webp`, sorted by filename) |
| `--frame-idx` | `None` | 0-based index into the sorted image list — process only that frame; default: all images (a single-file input is a 1-element list, so only `0` is valid) |
| `--prompts` | **required** | One or more object categories (open-vocabulary text prompts, `nargs='+'`); all joined into one detection call per image |
| `--model-path` | `IDEA-Research/Rex-Omni` | Hugging Face model id or local dir (downloaded to the HF cache on first use) |
| `--out_dir` | `./output/rexomni` | Output folder |

**Output** — flat under `--out_dir`, one PNG per input image (`<stem>.png`,
full input resolution, drawn with `RexOmniVisualize`): per-category boxes
with category labels (per-category colours, `font_size=20`, `draw_width=5`).
No file is written when the model detects nothing on that image; the
per-image console log reports each parsed category's object count and the
`type` + coordinates of every annotation. Unlike `infer_sam3.py` there is
no `--conf` flag — Rex-Omni's parser returns boxes without scores.

Key points:

- `inference()` returns one result dict per image with `success`,
  `extracted_predictions` (parsed by category: `{"type": "box"|"point"|...,
  "coords": [...]}` — boxes are absolute pixels `[x0, y0, x1, y1]`),
  `raw_output`, token counts and `image_size`.
- `--prompts` entries map 1:1 to the `categories` argument of
  `RexOmniWrapper.inference`; the model may echo a category back under a
  slightly different name, so the console log prints the **parsed** keys.

## Wrapper API

`RexOmniWrapper(model_path, backend=..., ...)` — `backend` is
`"transformers"` (default) or `"vllm"`. Generation is controlled by
`max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty` and
`stop`.

`inference(images, task, categories=..., keypoint_type=...,
visual_prompt_boxes=...)` — one image or a batch; `task` can be a single task
or a per-image list:

| Task | Output |
|---|---|
| `detection` | bounding boxes `[x0, y0, x1, y1]` per category |
| `pointing` | a point per object |
| `visual_prompting` | boxes of objects similar to the reference boxes (`visual_prompt_boxes`) |
| `keypoint` | boxes + keypoint coordinates (`keypoint_type`: `"person"` / `"hand"` / `"animal"`) |
| `ocr_box` / `ocr_polygon` | text detection + recognition |
| `gui_grounding` / `gui_pointing` | GUI element boxes / points |

`categories` is a single string, a list (applied to all images), or a list of
lists (per-image categories). See the `inference` docstring for batch
examples.

## Usage in the pipeline

Rex-Omni is the **object-detection stage of Step 3 (Sampling Keypoints)** of
the repo README pipeline: for each subtask, the text prompts (the
manipulator/object of the subtask's row in the dataset's
`meta/subtasks.csv`, e.g. `["brown cup", "left robot arm's grippers"]`)
drive the detection on the key-frames (start / end frames included). The
resulting bounding boxes, together with their text prompts, are passed to
SAM3 to segment the object masks (see [`sam3.md`](sam3.md)). Folder-mode
runs without a dataset pass the prompts explicitly with `--text-prompts`.
