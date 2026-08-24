"""
Standalone runtime for the exported SAM3 image model (torch.export .pt2).
"""

import os
from typing import Optional, Sequence, Union

import torch

# Register the C++ op the exported graph references before torch.export.load;
# without it deserialization fails with "failed to resolve
from torchvision.transforms import v2

# turn on tfloat32 for Ampere GPUs (mirrors sam3/model_builder.py)
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from . import utils
from .tokenizer import SimpleTokenizer
from utils.image_io import ImageInput, to_image_tensor

RESOLUTION = 1008

# Fixed number of box-prompt slots in the exported graph. Callers right-pad
# with mask=True for unused slots (see build_box_inputs).
BOXES_MAX = 8

_DEFAULT_BPE_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "bpe_simple_vocab_16e6.txt.gz"
)


class Sam3Image:
    """Off-the-shelf inference over an exported SAM3 image-model program.

    Args:
        checkpoint_path: .pt2 produced by test/export_image_model.py.
        bpe_path: tokenizer vocab (default: the vocab bundled with this
            package).
        device: torch device for inputs and outputs.
        autocast_dtype: dtype the program was traced with — bf16 (default) or
            fp16; the program must run under the matching outer autocast.

    Usage:
        model = Sam3Image("weights/sam3_image_exported_bf16.pt2")
        state = model.predict("image.jpg", text_prompt="visual",
                              boxes=norm_cxcywh, labels=[False, True])
    """

    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike],
        bpe_path: Optional[Union[str, os.PathLike]] = None,
        device: str = "cuda",
        autocast_dtype: Optional[torch.dtype] = None,
    ):
        if autocast_dtype is None:
            autocast_dtype = torch.bfloat16
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path or _DEFAULT_BPE_PATH)
        self.exported = torch.export.load(checkpoint_path)
        self.transform = v2.Compose(
                [
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(size=(RESOLUTION, RESOLUTION)),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def tokenize(self, prompt: str, context_length: int = 32) -> torch.Tensor:
        """Tokenize a text prompt into the program's [1, context_length]
        long-tensor input."""
        return self.tokenizer(prompt, context_length=context_length).to(self.device)

    def run(
        self,
        image_tensor: torch.Tensor,
        tokens: torch.Tensor,
        boxes: torch.Tensor,
        box_labels: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the exported program on the five raw graph inputs (see module
        docstring). Returns (pred_boxes, pred_logits, pred_masks,
        presence_logit_dec) under no_grad and the trace-matching autocast."""
        with torch.inference_mode(), torch.autocast(
            self.device, dtype=self.autocast_dtype
        ):
            return self.exported.module()(
                image_tensor, tokens, boxes, box_labels, box_mask
            )

    def preprocess_image(
        self, image: ImageInput, device: str = "cuda"
    ) -> torch.Tensor:
        """Same transform chain as Sam3Processor.set_image.

        Accepts a path, a PIL image, an RGB numpy array (HWC, uint8) or a
        CHW tensor (uint8 [0, 255] or float [0, 1]); returns the
        [1, 3, 1008, 1008] tensor normalized to [-1, 1].
        """

        input = to_image_tensor(image)             # uint8 CHW [0, 255] (or float CHW)
        return self.transform(input.to(device)).unsqueeze(0)

    def predict(
        self,
        image: ImageInput,
        text_prompt: str = "visual",
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[Sequence[bool]] = None,
        confidence_threshold: float = 0.5,
    ) -> dict[str, Union[torch.Tensor, int]]:
        """End-to-end prediction: image (path, PIL image, RGB numpy HWC or
        CHW tensor), an optional text prompt and optional box prompts
        (normalized cxcywh, labels True = positive) — returns the
        processor-style inference_state dict (boxes XYXY px, binarized masks,
        mask logits, scores, original size).

        Box-only prompting mirrors Sam3Processor.add_geometric_prompt's
        auto-set "visual" text prompt: pass boxes without text_prompt.
        """
        if (boxes is None) != (labels is None):
            raise ValueError("boxes and labels must be provided together")

        image_t = to_image_tensor(image)           # single conversion, reused below
        height, width = image_t.shape[-2:]
        image_tensor = self.preprocess_image(image_t, self.device)
        tokens = self.tokenize(text_prompt)
        if boxes is None:
            boxes_t = torch.zeros(BOXES_MAX, 4, device=self.device)
            box_labels_t = torch.zeros(BOXES_MAX, device=self.device, dtype=torch.long)
            box_mask = torch.ones(BOXES_MAX, device=self.device, dtype=torch.bool)
        else:
            boxes_t, box_labels_t, box_mask = self.build_box_inputs(
                boxes, labels, self.device
            )
        raw = self.run(image_tensor, tokens, boxes_t, box_labels_t, box_mask)
        with torch.autocast(self.device, dtype=self.autocast_dtype):
            out_boxes, out_masks, out_scores = self.postprocess(
                raw[0],
                raw[1],
                raw[2],
                raw[3],
                height,
                width,
                self.device,
                confidence_threshold,
            )
        return {
            "boxes": out_boxes,
            "masks": out_masks > 0.5,
            "masks_logits": out_masks,
            "scores": out_scores,
            "original_height": height,
            "original_width": width,
        }

    def build_box_inputs(
        self,
        norm_boxes: torch.Tensor,
        labels: Union[Sequence[bool], torch.Tensor],
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack normalized cxcywh boxes into the exported program's box-prompt
        inputs (BOXES_MAX fixed slots, valid boxes first, True = ignore)."""
        k = norm_boxes.shape[0]
        assert k <= BOXES_MAX, f"at most {BOXES_MAX} box prompts supported"
        boxes = torch.zeros(BOXES_MAX, 4, device=device)
        box_labels = torch.zeros(BOXES_MAX, device=device, dtype=torch.long)
        box_mask = torch.ones(BOXES_MAX, device=device, dtype=torch.bool)
        boxes[:k] = norm_boxes.to(device)
        box_labels[:k] = torch.tensor(labels, device=device, dtype=torch.long)
        box_mask[:k] = False
        return boxes, box_labels, box_mask

    def postprocess(
        self,
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        presence_logit_dec: torch.Tensor,
        orig_h: int,
        orig_w: int,
        device: str,
        confidence_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mirror of Sam3Processor._forward_grounding: score fusion, thresholding,
        cxcywh->xyxy + pixel scaling, mask interpolation.

        Returns (boxes XYXY px, masks [N, 1, H, W] probabilities, scores). Run
        under the program's autocast dtype for exact dtype parity.
        """
        out_probs = pred_logits.sigmoid()
        presence_score = presence_logit_dec.sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        out_masks = pred_masks[keep]
        out_bbox = pred_boxes[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = utils.box_cxcywh_to_xyxy(out_bbox)

        scale_fct = torch.tensor([orig_w, orig_h, orig_w, orig_h]).to(device)
        boxes = boxes * scale_fct[None, :]

        out_masks = utils.interpolate(
            out_masks.unsqueeze(1),
            (orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        return boxes, out_masks, out_probs
