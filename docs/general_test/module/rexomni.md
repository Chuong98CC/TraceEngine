# Rex-Omni Open-Vocabulary Detection (`tools/general_test/test_rexomni.py`)

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
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/test_rexomni.py
```

Do not install these packages into the main environment.

## Example script

`test_rexomni.py` is a **notebook-style example** rather than a configurable
CLI: the image path, categories and generation parameters are hard-coded at
the top. It detects the Astribot scene objects on the head camera frame
(`assets/astribot_test_imgs/head_rgbd/color/img_000000.jpg`):

```python
from det_seg_models.rex_omni import RexOmniWrapper, RexOmniVisualize

model = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",  # or "vllm" for faster inference
    max_tokens=4096, temperature=0.0, top_p=0.05, top_k=1,
    repetition_penalty=1.05,
)

results = model.inference(
    images=image, task="detection",
    categories=["left robot gripper", "right robot gripper", "brown coffee cup",
                "coffee machine", "transparent plastic tray",
                "dark green container"],
)
```

`results` is a per-image list of dicts with `success`, `extracted_predictions`
(parsed by category), `raw_output`, `inference_time`, token counts and
`image_size`. `RexOmniVisualize(image, predictions, font_size=20,
draw_width=5, show_labels=True)` draws the boxes and saves
`cache/rexomni_detection.jpg`.

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
the repo README pipeline: for each subtask, a text prompt is extracted from
the subtask description for each interacted object, and Rex-Omni detects the
objects on the key-frames (start / end frames included). The resulting
bounding boxes, together with their text prompts, are passed to SAM3 to
segment the object masks (see [`sam3.md`](sam3.md)).
