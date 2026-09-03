"""Rex-Omni open-vocabulary object detection over a folder of images.

Runs the Hugging Face Rex-Omni model (Qwen2.5-VL based — NOT a torch.export
checkpoint, unlike the other models in this repo) on each image of an input
folder — or on a single image file, or on one image of a folder selected by
index into the sorted file list — for the object categories given on the
command line, and saves **one annotated PNG per input image**: all detected
boxes of all categories drawn on the image (per-category colours + labels).

The categories are open-vocabulary text prompts and are joined into a single
detection call per image (`inference(task="detection", categories=...)`).
Rex-Omni is a generative model: its parser returns boxes per category with
**no per-box confidence scores**, so there is no ``--conf`` threshold, and
the model runs in the separate ``.venv-rexomni`` environment (Python 3.10,
torch 2.7 / transformers 4.51.3 — conflicts with the main env). Run this
tool with:

  PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
      -i assets/astribot_test_imgs/head_rgbd/color \
      --prompts "brown cup" "coffee machine" \
      --out_dir ./output/rexomni

  # single image file
  PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
      -i assets/astribot_test_imgs/head_rgbd/color/img_000000.jpg \
      --prompts "brown cup" --out_dir ./output/rexomni

  # one frame of a folder: frame-idx is a 0-based index into the sorted
  # file list (not a frame number)
  PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
      -i assets/astribot_test_imgs/head_rgbd/color --frame-idx 0 \
      --prompts "brown cup" --out_dir ./output/rexomni
"""

import argparse
from pathlib import Path

from PIL import Image

from det_seg_models.rex_omni import RexOmniVisualize, RexOmniWrapper

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rex-Omni open-vocabulary object detection on a folder "
        "of images -> one annotated PNG per input image"
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
        help="One or more object categories (open-vocabulary text prompts); "
             "all categories are joined into one detection call per image.",
    )
    parser.add_argument(
        "--model-path", type=str, default="IDEA-Research/Rex-Omni",
        help="Hugging Face model id or local dir (downloaded on first use).",
    )
    parser.add_argument("--out_dir", type=str, default="./output/rexomni")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = [p.strip() for p in args.prompts if p.strip()]
    if not prompts:
        raise SystemExit("--prompts needs at least one non-empty prompt")

    model = RexOmniWrapper(model_path=args.model_path, backend="transformers")
    print(f"[model] {args.model_path} (transformers backend)")

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
        result = model.inference(
            images=image, task="detection", categories=prompts
        )[0]
        if not result["success"]:
            print(f"  inference failed: {result.get('error', 'unknown')}")
            continue
        predictions = result["extracted_predictions"]  # {category: [anns]}
        for i, (category, annotations) in enumerate(predictions.items()):
            print(f"  [{i}] '{category}': {len(annotations)} object(s)")
            for j, ann in enumerate(annotations):
                coords = ", ".join(f"{v:.1f}" for v in ann["coords"])
                print(f"      id={j} type={ann['type']} coords=[{coords}]")
        if not predictions:
            print("  no object found for any prompt; nothing saved")
            continue
        vis_image = RexOmniVisualize(
            image=image, predictions=predictions, font_size=20, draw_width=5,
            show_labels=True,
        )
        out_path = out_dir / f"{image_path.stem}.png"
        vis_image.save(out_path)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
