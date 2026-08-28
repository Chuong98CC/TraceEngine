"""
Visualization helpers vendored from sam3/visualization_utils.py.

Only the functions test/base_box.py uses are included, so the heavy deps of
the original module (cv2, pandas, pycocotools, scikit-image, sklearn, tqdm)
are not needed — just matplotlib, numpy, PIL, torch. COLORS is generated with
a deterministic colorsys palette instead of the original sklearn-KMeans +
skimage-Lab based generation (same purpose: distinct per-object colors).
"""

import colorsys
from typing import Any, Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from PIL import Image


def generate_colors(n_colors: int = 128) -> np.ndarray:
    """Deterministic distinct-color palette (replaces sam3's sklearn/skimage
    based generator)."""
    colors = []
    for i in range(n_colors):
        hue = (i * 0.618033988749895) % 1.0  # golden-ratio spread
        sat = 0.5 + 0.25 * ((i // 4) % 4) / 3
        val = 0.55 + 0.25 * ((i // 16) % 3) / 2
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    return np.clip(np.asarray(colors), 0, 1)


COLORS = generate_colors(n_colors=128)

ImageInput = Image.Image | np.ndarray | torch.Tensor


def to_pil(img: ImageInput) -> Image.Image:
    """Normalize a PIL / numpy / torch image into a PIL image.

    Accepts PIL images, numpy arrays and torch tensors in HWC or CHW layout,
    as uint8 or as float in [0, 1] (scaled to uint8). Tensors are detached
    and moved off-device.
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dtype == torch.bfloat16:
            img = img.float()  # numpy cannot represent bfloat16
        img = img.numpy()
    else:
        img = np.asarray(img)
    if img.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {img.shape}; expected HWC, CHW or HW")
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = img.transpose(1, 2, 0)  # CHW -> HWC
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[..., 0]
    if img.dtype != np.uint8:
        if img.dtype.kind == "f" and img.size and img.max() <= 1.0:
            img = (img * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return Image.fromarray(img)


def draw_box_on_image(
    image: ImageInput,
    box: Sequence[float],
    color: tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """
    Draws a rectangle on a given image using the provided box coordinates in xywh format.
    :param image: PIL.Image, numpy.ndarray or torch.Tensor - The image on which to draw the rectangle.
    :param box: tuple - A tuple (x, y, x2, y2) representing the top-left and bottom-right corners of the rectangle.
    :param color: tuple - A tuple (R, G, B) representing the color of the rectangle. Default is green.
    :return: PIL.Image - The image with the rectangle drawn on it.
    """
    image = to_pil(image)
    # Ensure the image is in RGB mode
    image = image.convert("RGB")
    # Unpack the box coordinates
    x, y, x2, y2 = box
    x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
    # Get the pixel data
    pixels = image.load()
    # Draw the top and bottom edges
    for i in range(x, x2):
        pixels[i, y] = color
        pixels[i, y2 - 1] = color
        pixels[i, y + 1] = color
        pixels[i, y2] = color
        pixels[i, y - 1] = color
        pixels[i, y2 - 2] = color
    # Draw the left and right edges
    for j in range(y, y2):
        pixels[x, j] = color
        pixels[x + 1, j] = color
        pixels[x - 1, j] = color
        pixels[x2 - 1, j] = color
        pixels[x2, j] = color
        pixels[x2 - 2, j] = color
    return image


def normalize_bbox(
    bbox_xywh: list[float] | torch.Tensor, img_w: int, img_h: int
) -> list[float] | torch.Tensor:
    # Assumes bbox_xywh is in XYWH format
    if isinstance(bbox_xywh, list):
        assert len(bbox_xywh) == 4, (
            "bbox_xywh list must have 4 elements. Batching not support except for torch tensors."
        )
        normalized_bbox = bbox_xywh.copy()
        normalized_bbox[0] /= img_w
        normalized_bbox[1] /= img_h
        normalized_bbox[2] /= img_w
        normalized_bbox[3] /= img_h
    else:
        assert isinstance(bbox_xywh, torch.Tensor), (
            "Only torch tensors are supported for batching."
        )
        normalized_bbox = bbox_xywh.clone()
        assert normalized_bbox.size(-1) == 4, (
            "bbox_xywh tensor must have last dimension of size 4."
        )
        normalized_bbox[..., 0] /= img_w
        normalized_bbox[..., 1] /= img_h
        normalized_bbox[..., 2] /= img_w
        normalized_bbox[..., 3] /= img_h
    return normalized_bbox


def plot_bbox(
    img_height: int,
    img_width: int,
    box: Sequence[float],
    box_format: str = "XYXY",
    relative_coords: bool = True,
    color: str = "r",
    linestyle: str = "solid",
    text: str | None = None,
    ax: Axes | None = None,
) -> None:
    # matplotlib converts box coords via np.asarray, which torch tensors
    # (especially bfloat16) do not support; normalize to Python floats.
    box = [v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for v in box]
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )


def plot_mask(
    mask: np.ndarray | torch.Tensor,
    color: str = "r",
    ax: Axes | None = None,
) -> None:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().float().numpy()  # bool/bf16 -> float32 numpy
    else:
        mask = np.asarray(mask)
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)


def plot_results(img: ImageInput, results: dict[str, Any]) -> None:
    img = to_pil(img)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
