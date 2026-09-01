# RoMAv2 Multi-Image Keypoint Matching (`tools/general_test/module/infer_romav2.py`)

Finds matching keypoints **visible in all given images** with the RoMAv2
dense-matching model, exported as a `torch.export` program
(`weights/romav2/romav2.pt2`). The `RoMaV2PT2` wrapper
(`src/det_seg_models/romav2/romav2.py`) only depends on torch / numpy / PIL —
no `romav2` package, no checkpoint download.

Points are anchored in the **first image** (balanced sampling from the (0, 1)
pair) and projected into every other image through that pair's dense warp
field.

```bash
# wrapper script: 4 sample key-frames of the coffee cup
bash scripts/general_test/infer_romav2.sh

# or directly — 2+ image paths
python tools/general_test/module/infer_romav2.py \
    assets/matching_points/coffee1.png assets/matching_points/coffee2.png \
    assets/matching_points/coffee3.png assets/matching_points/coffee4.png \
    --strategy reference --num-corresp 2000 --top-k 128 \
    --model weights/romav2/romav2.pt2 \
    --out cache/multi_match_matches.npz --viz cache/multi_match_vis.jpg
```

With 2 images this reduces to the plain pairwise case: one match call, and
the overlap filter is the only difference from `model.sample()` output.

## Matching strategies

| `--strategy` | Behavior |
|---|---|
| `reference` (default) | match image 0 against every other image (N−1 calls); positions come straight from each pair's warp field |
| `cycle` | additionally match all remaining pairs (N(N−1)/2 calls total) and keep only points whose tracked positions agree with every pair's warp in **both** directions (round-trip error below `--cycle-err`) |

## Point selection (choose exactly one)

| Argument | Method |
|---|---|
| `--overlap-th <th>` (default `0.25`) | keep every candidate whose overlap confidence exceeds `th` in **all** images |
| `--top-k <k>` | keep the `k` best candidates ranked by worst-case overlap across images (min over images), ignoring `--overlap-th` |

In `cycle` mode both selections first drop round-trip-inconsistent candidates
(`--cycle-err` is a hard filter, not a ranking criterion; default `0.01` in
normalized coordinates). When fewer than `top-k` candidates survive, all of
them are kept.

## Arguments

| Argument | Default | Description |
|---|---|---|
| `images` | **required** | 2+ image paths; matching points are anchored in the first image |
| `--strategy` | `reference` | `reference` or `cycle` (see above) |
| `--num-corresp` | `500` | candidate points sampled in image 0 before filtering |
| `--overlap-th` | `0.25` | threshold method (not allowed with `--top-k`) |
| `--top-k` | — | top-k method (not allowed with `--overlap-th`) |
| `--cycle-err` | `0.01` | max round-trip warp error in normalized coords, cycle mode |
| `--model` | **required** | path to the exported `.pt2` program (the `RoMaV2PT2` class itself defaults to `weights/romav2/romav2.pt2`) |
| `--out` | **required** | where to save the matches as `.npz` |
| `--viz` | **required** | where to save the side-by-side visualization |
| `--no-viz` | off | skip saving the visualization |

## Output

| File | Content |
|---|---|
| `<out>.npz` | `matches` (K, N, 2) — pixel coordinates of the K surviving points in each of the N images; `image_paths` |
| `<viz>.jpg` | the matches drawn on the images stacked side-by-side (all resized to image 0), one color per point |

## Notes

- The exported program is fixed-resolution: `romav2.json` next to the `.pt2`
  stores `H_lr` / `W_lr`, and `match_pair` bicubic-resizes inputs to it.
- The graph always computes **both directions** (warp / confidence / overlap /
  precision for A→B and B→A); `match_pair` returns all eight tensors.
- The `.pt2` stores tensors on the device it was exported from (CUDA here) —
  running on a CPU-only machine requires a CPU export.

## Usage in the pipeline

RoMAv2 is the **keypoint-matching stage of Step 3 (Sampling Keypoints)** of
the repo README pipeline: for each subtask, the objects detected by RexOmni
and segmented by SAM3 on the key-frames are cropped (bounding box enlarged by
a scale), and `infer_romav2.py` finds the keypoints that match consistently
**across the key-frames** of the subtask. Only the top-k keypoints that fall
inside the object masks are kept; TAPIP3D then tracks them over the whole
subtask (Step 4, see [`tapip3d.md`](tapip3d.md)).
