"""
Run the exported SAM3 image model on box and text prompts using only
(no sam3 import) — the standalone twin of test/base_exported.py, which is
itself the exported twin of test/base_box.py.

Usage:
    python sam3_runtime/example.py          # or: python -m sam3_runtime.example
"""
#%%
# # Make the package importable when run as a plain script from anywhere.
# import os
# import sys
#
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#%%
import matplotlib.pyplot as plt
import torch
from PIL import Image
# turn on tfloat32 for Ampere GPUs (also done by sam3_runtime on import)
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire script (the model class also wraps its runs and
# post-processing in the trace-matching autocast)
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from det_seg_models.sam3 import (
    Sam3Image,
    draw_box_on_image,
    normalize_bbox,
    plot_results,
)

from det_seg_models.sam3.utils import box_xyxy_to_cxcywh

model = Sam3Image("../../weights/sam3/sam3_image_exported_bf16.pt2")

#%%
# infer image path
image_path = "/data/astri_making_coffee_v1/eps_data/key_frames/ep000000/subtask_00/cam_head/frame_000000.jpg"
image = Image.open(image_path)
width, height = image.size

#%%
# infer box
box_input_xyxy = [[1, 240, 100.0, 340.0]]
box_input_cxcywh = box_xyxy_to_cxcywh(torch.tensor(box_input_xyxy).view(-1,4))
norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height)

box_labels = [True]

img0 = Image.open(image_path)
image_with_box = img0
for i in range(len(box_input_xyxy)):
    if box_labels[i] == 1:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)
    image_with_box = draw_box_on_image(image_with_box, box_input_xyxy[i], color)
plt.imshow(image_with_box)
plt.axis("off")  # Hide the axis
plt.show()

#%%
# Box-only prompting: Sam3Processor.add_geometric_prompt auto-sets the
# "visual" text prompt when no text is set — predict() handles preprocessing,
# inference and post-processing end-to-end.
inference_state = model.predict(
    image,
    text_prompt="brown cup",
    boxes=norm_boxes_cxcywh,
    labels=box_labels,
)

plot_results(img0, inference_state)

#%%
# Text prompt on the same image, after the box prompts (the notebook's
text_inference_state = model.predict(image, text_prompt="brown cup")

# Get the masks, bounding boxes, and scores
print(text_inference_state["boxes"])
print(text_inference_state["scores"])

img1 = Image.open(image_path)
plot_results(img1, text_inference_state)
# %%
