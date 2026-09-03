"""SAM3 text-prompt segmentation over a folder of images via the torch.export
(PT2) runtime.

Runs the exported SAM3 image model (fixed 1008 resolution, fixed prompt
slots) on each image of an input folder — or on a single image file, or on
one image of a folder selected by index into the sorted file list — for
every text prompt given on the command line, and saves **one overlay PNG
per input image**: all prompts' detected instances drawn together on the
image (coloured masks + boxes + scores, label ``[p] id=i, score`` where
``p`` is the prompt index).

Examples:
  python tools/general_test/module/infer_sam3.py \
      -i assets/astribot_test_imgs/head_stereo_left \
      --prompts "brown cup" "coffee machine" \
      --out_dir ./output/sam3

  # single image file
  python tools/general_test/module/infer_sam3.py \
      -i assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg \
      --prompts "brown cup" --out_dir ./output/sam3

  # one frame of a folder: frame-idx is a 0-based index into the sorted
  # file list (not a frame number)
  python tools/general_test/module/infer_sam3.py \
      -i assets/astribot_test_imgs/head_stereo_left --frame-idx 2 \
      --prompts "brown cup" --out_dir ./output/sam3
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; set before pyplot is imported

import torch
from PIL import Image

from det_seg_models.sam3 import Sam3Image
from utils.visualize.visualize_mask import COLORS, plot_bbox, plot_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_path: Path) -> list[Path]:
    """Image paths to process from ``input_path``: the file itself when it is
    a single image, otherwise the sorted images of a folder (extension
    filter)."""
    if input_path.is_file():
        return [input_path]
    images = sorted(
        p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        exts = ", ".join(sorted(IMAGE_EXTS))
        raise FileNotFoundError(
            f"No image files ({exts}) found in {input_path}"
        )
    return images


def save_overlay(
    image: Image.Image,
    detections: list[tuple[int, dict]],
    out_path: Path,
) -> None:
    """Save one overlay per input image: all (prompt_idx, predict state)
    pairs drawn on the same full-pixel-size canvas."""
    import matplotlib.pyplot as plt

    width, height = image.size
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))  # axis-less full-figure canvas
    ax.axis("off")
    ax.imshow(image)
    # colour groups offset per prompt index so the same object id in
    # different prompts does not share a colour
    for prompt_idx, state in detections:
        for i in range(len(state["scores"])):
            color = COLORS[(prompt_idx * 16 + i) % len(COLORS)]
            plot_mask(state["masks"][i].squeeze(0).cpu(), color=color, ax=ax)
            plot_bbox(
                height,
                width,
                state["boxes"][i].cpu(),
                text=f"[{prompt_idx}] id={i}, {state['scores'][i].item():.2f}",
                box_format="XYXY",
                color=color,
                relative_coords=False,
                ax=ax,
            )
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 text-prompt segmentation on a folder of images "
        "-> one overlay PNG per input image"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input image folder (jpg/png/bmp/webp), or a single image path.",
    )
    parser.add_argument(
        "--frame-idx", type=int, default=None,
        help="0-based index into the sorted image list to process only that "
             "frame; default: process all images.",
    )
    parser.add_argument(
        "--prompts", nargs="+", required=True,
        help="One or more text prompts; each prompt runs its own predict() "
             "call and all detections are drawn on the image overlay.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Detection confidence threshold passed to predict().",
    )
    parser.add_argument(
        "--pt2", type=str, default="weights/sam3/sam3_image_exported_bf16.pt2",
        help="Path to the exported graph checkpoint.",
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (must be CUDA).")
    parser.add_argument("--out_dir", type=str, default="./output/sam3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [p.strip() for p in args.prompts if p.strip()]
    if not prompts:
        raise SystemExit("--prompts needs at least one non-empty prompt")

    # tfloat32 for Ampere GPUs (also done by the sam3 runtime on import)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = Sam3Image(args.pt2, device=args.device)
    print(f"[model] {args.pt2}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"No such file or directory: {input_path}")
    images = list_images(input_path)
    if args.frame_idx is not None:
        if not 0 <= args.frame_idx < len(images):
            raise SystemExit(
                f"--frame-idx {args.frame_idx} out of range: "
                f"{len(images)} image(s) from {input_path}"
            )
        images = [images[args.frame_idx]]
    print(f"[input] {len(images)} image(s) from {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        print(f"\n[{image_path.name}] size={image.size[0]}x{image.size[1]}")
        detections: list[tuple[int, dict]] = []
        for prompt_idx, prompt in enumerate(prompts):
            state = model.predict(
                image, text_prompt=prompt, confidence_threshold=args.conf
            )
            n_objects = len(state["scores"])
            print(f"  [{prompt_idx}] '{prompt}': {n_objects} object(s)")
            if n_objects == 0:
                continue
            for i in range(n_objects):
                box = [round(v, 1)
                       for v in state["boxes"][i].cpu().tolist()]
                print(f"      id={i} score={state['scores'][i].item():.2f} "
                      f"box_xyxy={box}")
            detections.append((prompt_idx, state))

        if not detections:
            print("  no object found for any prompt; nothing saved")
            continue
        out_path = out_dir / f"{image_path.stem}.png"
        save_overlay(image, detections, out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
