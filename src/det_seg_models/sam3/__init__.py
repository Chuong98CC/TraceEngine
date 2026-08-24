"""
Standalone runtime for the exported SAM3 image model — inference without
importing sam3.

Everything needed to run the torch.exported program from
test/export_image_model.py is vendored here: the BPE tokenizer, the image
preprocessing / box packing / postprocessing, and the visualization helpers.
Dependencies: torch, torchvision, ftfy, regex (+ matplotlib/PIL for viz).

Usage:
    from sam3_runtime import Sam3Image

    model = Sam3Image("weights/sam3_image_exported_bf16.pt2")
    state = model.predict("image.jpg", text_prompt="visual", boxes=..., labels=...)
"""

from .sam3_model import (
    BOXES_MAX,
    RESOLUTION,
    Sam3Image,
)
from .tokenizer import SimpleTokenizer
from utils.visualize_mask import draw_box_on_image, normalize_bbox, plot_results

__all__ = [
    "BOXES_MAX",
    "RESOLUTION",
    "Sam3Image",
    "SimpleTokenizer",
    "draw_box_on_image",
    "normalize_bbox",
    "plot_results",
]
