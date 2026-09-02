"""Minimal runnable example: preprocess -> inference -> post-processing.

Loads the exported graph + refiner companion weights, runs MoGe v3 on a
single image and prints the output tensors (no visualization, no mesh).

Usage:
    uv run python minimal_infer_pt2/demo.py -i demo_data/.../frame_000001.jpg
"""

import time
import argparse
from pathlib import Path


from depth_models.moge3.moge_pt2 import MoGev3_PT2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-i', '--input', required=True, help='Input image path (jpg/png).')
    parser.add_argument('--pt2', type=str, default='weights/moge3/moge3_l.pt2',
                        help='Path to the exported graph checkpoint.')
    parser.add_argument('--refiner', type=str, default=None,
                        help='Path to the refiner companion checkpoint (defaults to the .pt2 path with _refiner.pt).')
    parser.add_argument('--device', type=str, default='cuda', help='Device (must be CUDA).')
    parser.add_argument('--refine_steps', type=int, default=3, help='Sparse refinement steps.')
    args = parser.parse_args()

    model = MoGev3_PT2(args.pt2, refiner_path=args.refiner, device=args.device,
                       refine_steps=args.refine_steps)

    t0 = time.perf_counter()
    out = model.infer_file(args.input)
    elapsed = time.perf_counter() - t0
    print(f'inference took {elapsed * 1000:.1f} ms')
    for name, tensor in out.items():
        print(f'  {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}')


if __name__ == '__main__':
    main()
