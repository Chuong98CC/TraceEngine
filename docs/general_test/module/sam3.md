# Promptable Segmentation — SAM3 (`tools/general_test/module/infer_sam3.py`)

Runs the exported SAM3 image model (`sam3_image_exported_bf16.pt2`) on box
and text prompts. This script is a **notebook-style example** (`#%%` cells,
editable in any Python/Jupyter environment) rather than a configurable CLI:
the image path and prompt boxes are hard-coded at the top, and the model is
loaded with hard-coded relative paths (`../../weights/...`, run from
`tools/general_test/`).

```python
from det_seg_models.sam3 import Sam3Image

model = Sam3Image("../../weights/sam3/sam3_image_exported_bf16.pt2")

# Box prompts (xywh) + labels; predict() handles pre/infer/post end-to-end
inference_state = model.predict(image, text_prompt="visual",
                                boxes=norm_boxes_cxcywh, labels=[True, True])

# Text-only prompt
text_state = model.predict(image, text_prompt="brown cup")
```

Key points:

- `predict()` returns a state dict with `masks`, `boxes`, `scores`.
- Box prompts are normalized to cxcywh (`box_xywh_to_cxcywh` +
  `normalize_bbox`); `draw_box_on_image` / `plot_results` visualize.
- The graph is exported at a fixed 1008 resolution with a fixed number of
  box-prompt slots; callers right-pad prompts.

## Usage in the pipeline

SAM3 provides the **object prompt** for TAPIP3D 3D tracking:
`infer_tapip3d.py --bbox ... --text_prompt ...` segments the tracked object's
box on the first frame and samples the query points inside the mask (see
[`tapip3d.md`](tapip3d.md)).
