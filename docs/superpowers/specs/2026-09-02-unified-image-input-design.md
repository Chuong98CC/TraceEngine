# Unified image input handling across model runtimes

Date: 2026-09-02 · Status: revised per review, pending re-review

## 1. Goal

Every model runtime currently duplicates its own image read + input-type
dispatch (str path / PIL / numpy / tensor branching) and its own letterbox
geometry (cv2-based in DA3 + WAFT, PIL-based in VGGT), with two conflicting
channel conventions (DA3/WAFT assume **BGR** numpy; SAM3/RoMaV2/VGGT/Any2Full
assume **RGB**). Factor the shared work into `src/utils/image_io.py` so that:

1. every runtime accepts the full `ImageInput` union (str / Path / PIL /
   HWC-numpy / CHW-tensor),
2. image load + geometry run as **torch ops on CHW tensors** — every model is
   (or, for WAFT, will soon be) a `torch.export` program that must be fed
   `torch.Tensor`s, so preprocessing is **tensor-first**: decode → letterbox →
   normalize all in torch, and the exported graphs receive tensors directly.
3. every runtime's per-image entry method is uniformly named **`_load_image`**,
4. the whole repo uses one channel contract: **numpy/tensor images are RGB,
   uint8** (float `[0,1]` accepted from tensor sources and rescaled on entry).

numpy remains only where it is the native downstream domain: model **outputs**
(streaming saves npz; alignment/crop helpers and visualizers are numpy), and
depth maps/masks.

**WAFT is out of scope, untouched**: it will be replaced by a `torch.export`
`.pt2` runtime (and its ONNX/TensorRT backends removed) soon, and should adopt
this unified pipeline at that point. Until then it keeps its own BGR + numpy
preprocess — the last legacy dialect in the repo.

## 2. Recorded decisions

| Decision | Choice |
|---|---|
| Input acceptance scope | src/ runtime APIs + shared frame loaders in `utils/streaming_utils.py` (TAPIP3D `[T,3,H,W]` batches). Not the visualize/export tools, not RexOmni (separate env), not legacy `MonoDepthTRT`/`StereoDepthTRT` (zero callers), not depth-map geometry (`a2fnested._place_on_grid`, depth resizers). |
| Channel contract | RGB everywhere **except WAFT, which is untouched** (keeps `bgr_input` until its planned `.pt2` replacement). BGR sources among the converted runtimes (online-streaming cache) flip once at the source. numpy arrays must be uint8 (float numpy dropped — was only supported by RoMaV2). |
| Input representation | **Tensor-first**: `torch.export` runtimes receive `torch.Tensor`s straight from preprocess — no numpy bounce at the feed boundary. Model outputs stay numpy. (WAFT, untouched, still feeds numpy to its ONNX/TRT backends.) |
| Loader method name | Every image-ingesting runtime class gets `_load_image(self, image: ImageInput) -> torch.Tensor` (CHW uint8 RGB, decode-only, on CPU). Per-runtime float scaling/batching/device moves to the preprocess callers. |
| Geometry depth | Full tensor-space rewrite of image read + letterbox/normalize via a small **family of shared helpers** in `image_io.py`; per-model normalize/resize chains that are genuinely model-specific stay. |
| Letterbox shape | One shared `letterbox()` with mode knobs preserving each model's geometry convention (DA3/WAFT `trunc2` + bilinear; VGGT `round` + bicubic). |
| Unit tests | Add a minimal CPU-only `tests/test_image_io.py`; pytest added as a dev dependency group. |
| Golden verification | Committed harness script (`tools/general_test/module/compare_image_inputs.py` + `scripts/general_test/verify_image_inputs.sh`) capturing pre-refactor goldens, re-run post-refactor. |

## 3. Target API — `src/utils/image_io.py`

Module docstring updated to state the RGB/uint8 contract. Exports:

```python
# unchanged
def to_image_tensor(image: ImageInput, mode: Optional[str] = "RGB") -> torch.Tensor
    # str/PIL/numpy -> CHW uint8; tensor passthrough (uint8 or float [0,1]);

def letterbox(x: torch.Tensor, target_h: int, target_w: int, *,
              scale_mode: Literal["trunc2", "round"] = "trunc2",
              resize: InterpolationMode = InterpolationMode.BILINEAR,
              antialias: bool = False) -> tuple[torch.Tensor, dict]:
    """Aspect-preserving center-pad of a CHW image.

    x: CHW pixel-space (uint8, or float [0,1] accepted and rescaled to uint8
       on entry; C must be 3 — clear ValueError otherwise).
    Scale math in float64 python (bit-identical to today's numpy arithmetic):
      trunc2: raw = min(tw/w0, th/h0); scale = floor(raw*100)/100, fallback
              raw when <= 0; new_w/h = int(...)         (DA3 / WAFT convention)
      round : scale = min(...); new_w/h = round(...)    (VGGT convention)
    resize via torchvision.functional.resize on uint8 (keeps the integer
    quantization the models were trained with); zero-pad via F.pad centered.
    Returns (padded uint8 CHW, meta) with today's exact keys: orig_h, orig_w,
    scale_factor (float), tile_h, tile_w, pad_top, pad_left — bit-identical
    to the current cv2/PIL meta dicts."""

def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """(x/255 - mu)/std in fp32, ImageNet constants — currently duplicated as
    BaseDA3Model._MEAN/_STD (numpy) and the T.Normalize tuple in any2full."""
```

All three are pure torch — no numpy in the geometry.

## 4. Unified `_load_image` contract

Every image-ingesting wrapper implements:

```python
def _load_image(self, image: ImageInput) -> torch.Tensor:
    """Decode any ImageInput into a CHW uint8 RGB tensor (CPU)."""
```

Body: `to_image_tensor(image)`; float `[0,1]` CHW tensor entries are rescaled
`(x.clamp(0,1)*255).round().byte()` (documented — decode-space is uint8, like
reading a file). Everything a runtime additionally needs (float/255,
ImageNet norm, batching, device) happens in the preprocess caller, outside
`_load_image`.

- `BaseDA3Model._load_image` (new; replaces the inline read in `_preprocess_one`)
- `BaseVGGTOmega._load_image` (new; replaces module fn `_load_rgb_uint8`)
- `RoMaV2PT2._load_image` (existing name; becomes decode-only — its
  float/255 + `[None]` + device steps move into `match_pair`)
- `Sam3Image._load_image` (new thin wrapper; transform stays in
  `preprocess_image`)
- `Any2Full_PT2._load_image` (new; replaces module fn `_load_rgb`, whose
  normalize step moves into `preprocess`)

(WAFTBase keeps its existing `_load_image` exactly as-is — out of scope.)

## 5. Per-file changes

### 5.1 `src/depth_models/da3/model/` — DA3 (tensor-first)

`base_da3.py`:
- Delete `_MEAN`/`_STD` and the lazy `cv2` import; add `_load_image` (see §4).
- `_preprocess_one(img: ImageInput, target_h, target_w)`: `_load_image` →
  `letterbox(scale_mode="trunc2")` → `imagenet_normalize` — returns a
  **fp32 CHW tensor** (values identical to today's numpy result).
- `preprocess_views(imgs)`: stacks into a **CPU fp32 tensor `(1,N,3,H,W)`**
  (was numpy) + unchanged meta list. Docstrings: numpy arrays are RGB.
- `crop_to_tile`/key-mapping/sky/alignment: unchanged (numpy outputs).

`da3anyview.py` — `run(img_batch: torch.Tensor)`: drop
`torch.from_numpy(...).astype(np.float32)` — `img_batch.to(self.device)`
directly. Outputs stay numpy (`.float().cpu().numpy()`). `infer` unchanged.

`da3metric.py` — `run(feed)`/`infer_view(img: torch.Tensor CHW)`: accept a
tensor; drop the numpy cast/`from_numpy`. Outputs stay numpy.

`da3nested.py` — `_run_metric_branch(av_img_batch: torch.Tensor)`: slicing
`av_img_batch[0, i]` yields the tensor view → `metric.infer_view(tensor)`.
Depth/sky accumulation stays numpy (alignment contract).

Internal-only API change: no external callers of `preprocess_views`/`run`
exist outside these wrappers (streaming calls `model.infer(paths)`).

### 5.2 `src/depth_models/vggt_omega/base_vggt_omega.py` + `vggt_omega.py`

- Delete module fns `_load_rgb_uint8`, `_letterbox_preprocess`; add
  `BaseVGGTOmega._load_image` (§4).
- `infer(images: list[ImageInput])`: per frame `_load_image` →
  `letterbox(scale_mode="round", resize=BICUBIC)` → `.float()/255.0` (CHW
  float `[0,1]` canvas, replacing the old HWC float canvas that only existed
  for the PIL paste). `build_feed` returns `{input_name: tensor}` —
  `torch.stack(canvases)[None]` — a CPU fp32 `(1,N,3,H,W)` tensor (was
  numpy).
- `vggt_omega.py._forward(feed)`: accept the tensor feed; keep the graph
  dtype cast; `.to(self.device)` instead of `torch.from_numpy(...).cuda()`.
  Outputs stay numpy.

### 5.3 `src/det_seg_models/romav2/romav2.py`

- `_load_image` becomes decode-only (§4); its float/255 + `[None]` + device
  steps move to `match_pair` (both images) — `match` unchanged.
- Drop float-numpy input support (uint8 contract); keep the explicit `I;16`
  `NotImplementedError` guard (checked on PIL images/paths).
- Import `ImageInput` from `utils.image_io`; delete the local alias.
  Docstring updated ("standalone" note: adds torchvision dependency via
  `utils.image_io` — same main env, fine).

### 5.4 `src/depth_models/a3f/any2full.py`

- Module fn `_load_rgb` becomes instance `_load_image` (decode-only, §4).
- `preprocess(self, rgb: ImageInput, depth_metrics, ...)`: parameter renamed
  from `rgb_path`; body: `_load_image` → float/255 → shared
  `imagenet_normalize` (replaces the local `T.Normalize` tuple) → unsqueeze →
  device/dtype → `_check_and_resize` (unchanged). Same op order/values.
- Callers (`infer_any2full.py` passes the rgb path; `a2fnested` passes
  arrays) need no logic changes — only type-hint/docstring touch-ups.

### 5.5 `src/det_seg_models/sam3/sam3_model.py`

- Add decode-only `_load_image` (§4); `preprocess_image` calls it before the
  existing `self.transform` chain (already tensor-native, unchanged).
  Numerics identical.

### 5.6 `src/utils/streaming_utils.py` (TAPIP3D frame loaders)

- `load_batch_frames`: per frame `to_image_tensor` → returns a **CHW uint8
  CPU tensor stack `(T,3,H,W)`** (was `(T,H,W,3)` numpy). Its only caller is
  `load_resized_batch`.
- `load_resized_batch`: resize via `torchvision.functional.resize` (bilinear,
  antialias=False) on the CHW tensor → float `/255` → keeps its CPU-tensor
  output contract `(T,3,H,W)` float [0,1] (what the TAPIP3D `.pt2` consumes).
  Depth/geometry (`load_npz_batch`, `resize_depth_bilinear`) untouched.
- `load_pair`: path decode + resize-to-depth-shape via the image_io helpers;
  keeps its `(N,H,W,3)` uint8 numpy output — its only consumer is
  `visualize_stream.py` (numpy rendering domain).

### 5.7 `src/depth_models/a3f/a2fnested.py`

- `infer(imgs: list[ImageInput], depths)`: docstring "paths or BGR arrays" →
  ImageInput RGB. `_run_metric_branch` keeps its `np.ascontiguousarray` guard
  (torch.from_numpy constraint — now only relevant for depth maps).
  **Latent bug fixed by this refactor**: the same RGB views now reach both
  the any-view grid and the Any2Full branch with consistent channels (today
  DA3 flips BGR→RGB while Any2Full treats the same BGR array as RGB).

### 5.8 `tools/astribot/run_step2_depth_stream.py` (online streaming)

Dataset frames are PIL RGB; the file flips them to BGR once at fetch only
because DA3/WAFT wanted BGR. With RGB everywhere for the converted runtimes:
store RGB (delete the BGR flip in the frame fetch), delete the `_rgb_input`
class flag and the VGGT flip-back branch. **WAFT exception**: the motion-mask
path still constructs `WAFT(..., bgr_input=True)` and feeds it BGR — flip
RGB→BGR (contiguous copy) at that call site only. Update the "frames are
stored BGR …" comments.

### 5.9 Not touched (explicit)

**WAFT** (`base_waft.py`, `waft.py`, `infer_waft.py` — kept exactly as-is),
`MonoDepthTRT`/`StereoDepthTRT` (`base_trt.py`, no callers), RexOmni (own
`.venv-rexomni` env), depth-map geometry (`a2fnested._place_on_grid`,
`a2f_streaming._load_depth`/probe, `base_streaming` mask reads,
`resize_depth_bilinear`, `video_io`, `visualize_*`, `astribot_dataloader` —
returns an RGB *path*, by design), `base_streaming` disk mode (passes paths,
inherits everything).

## 6. Numeric-drift expectation

| Stage | Old | New | Expected delta |
|---|---|---|---|
| path decode | cv2.imread (BGR→RGB) | PIL via to_image_tensor | bit-identical (verified) |
| resize | cv2 uint8 fixed-point / PIL BICUBIC uint8 | tvf.resize uint8 (antialias=False) | ≤ 1–3 LSBs, edges |
| scale/pad math | numpy float64 + cv2 copyMakeBorder | same float64 + F.pad | identical |
| normalize / /255 / stack | per model (numpy) | shared helpers (torch fp32) | identical op order — bit-identical expected, asserted by golden |
| feed boundary | torch.from_numpy + astype | direct tensor | none (same values) |

Everything downstream (engine/exported-graph runs) is a continuous function
of these inputs; deltas stay in the last-LSB band.

## 7. Verification plan

Repo has no tests today; verification is a real gate, in two layers.

### 7.1 Golden harness (committed)

`tools/general_test/module/compare_image_inputs.py` (`capture` / `compare`
subcommands) + `scripts/general_test/verify_image_inputs.sh` wrapper
(goldens written under `output/`, not committed):

- **capture** (against pre-refactor code): per runtime, feed real frames from
  `assets/astribot_test_imgs` and save npz: DA3 `preprocess_views` batch +
  meta (dummy subclass with target sizes from the checkpoint file names — no
  model load), VGGT letterbox canvases + crops, RoMaV2 loaded images, Any2Full
  `preprocess` tensors, SAM3 preprocess tensor (already canonical — smoke
  only), TAPIP3D `load_batch_frames` / `load_resized_batch` outputs. (WAFT
  excluded — untouched.) Optional `--e2e` additionally runs each real model
  end-to-end (all weights present locally) and saves depth/pose outputs.
- **compare** (post-refactor): diffs arrays/tensors (cpu) in float64. Pass
  criteria: meta keys/values **exact**; uint8-stage outputs ≥ 99.9 % pixels
  within 1 LSB (2 for VGGT bicubic), max ≤ 3; fp32 preprocess within 2e-2 abs
  after normalization (≈1 LSB propagated); e2e outputs reported
  (mean/max/percentile diffs) with a sanity bound, not a hard assert.

### 7.2 Unit tests (committed, CPU-only, no weights)

`tests/test_image_io.py` (pytest, new `dev` dependency group in pyproject):

- `to_image_tensor`: tmp-path png, PIL mode convert, RGB numpy vs manual CHW
  reference, non-writable numpy copy, tensor passthrough, error cases.
- `letterbox`: exact meta on known sizes (identity fit; trunc2 downscale;
  round mode; pad centering), uint8 dtype + zero padding, float [0,1] entry
  conversion, cross-check vs the old cv2 reference implementation on a random
  image (meta exact, pixels ≤ 1 LSB).
- `imagenet_normalize`: known-value spot checks + fp32.

Run line documented in CLAUDE.md: `uv run --group dev pytest tests/`.

## 8. Docs to update (same PR)

- `CLAUDE.md`: `image_io.py` tree line ("ImageInput, to_image_tensor,
  letterbox, imagenet_normalize"), the "Model input/output conventions"
  section (RGB-everywhere uint8 contract with the WAFT exception, ImageInput
  acceptance, tensor-first feeds, unified `_load_image`), any DA3/VGGT
  class-hierarchy lines mentioning BGR.
- Module docstrings of every touched file (§5). WAFT docs untouched.

## 9. Implementation order (for the plan)

1. Golden capture: add the harness, run `capture` (+ `--e2e`) on current code.
2. `image_io.py` helpers + `tests/test_image_io.py` (green against cv2/PIL
   references).
3. Runtime rewrites, one by one, each diffed against goldens immediately:
   DA3 (`base_da3` → `da3anyview` → `da3metric` → `da3nested`) → VGGT
   (base + wrapper) → RoMaV2 → SAM3 → Any2Full → `streaming_utils` →
   `a2fnested` hints → `run_step2_depth_stream.py`.
4. `compare` full pass + e2e, docs, CLAUDE.md.

## 10. Out of scope / follow-ups

- **WAFT**: untouched here; its planned `torch.export` `.pt2` replacement (and
  removal of the ONNX/TensorRT backends) should adopt this unified RGB +
  `_load_image` + tensor-first pipeline at that time, and then the
  RGB→BGR flip in `run_step2_depth_stream.py`'s motion-mask path disappears.
- Depth-map geometry still cv2 (`_place_on_grid`, depth resizers, mask reads)
  — candidate for a later unification with the same helpers once they accept
  NCHW depth.
- `MonoDepthTRT`/`StereoDepthTRT` are dead code — delete or revive separately.
- CLAUDE.md lists `tools/astribot/run_subtask_a3f.py` which no longer exists
  (doc drift, unrelated — flag, don't fix here).
