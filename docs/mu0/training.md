# μ₀ — training / fine-tuning / evaluation (`src/mu0`)

**μ₀** ([paper: arXiv:2606.13769](https://arxiv.org/abs/2606.13769), project
page: <https://mu0-wm.github.io>) is a **trace world model**: from
`(image, language, history of N keypoints)` it predicts the future 3D traces of
semantic interaction points (objects, tools, hands, contact regions) as
B-spline control points in an anchor-relative, normalized space (image-plane
`uv` + metric depth), conditioned on per-keypoint DINOv2 features, a
metric-depth channel, and a per-keypoint "done" head. It is built on LeRobot's
SmolVLA backbone.

This repo ports `github.com/Yoonkyo/mu0` @ `1a95186`
(`1a95186997fb30434d213a6e0b19cb43e8965e11`) into `src/mu0`, importing pip
`lerobot[dataset,training,smolvla]==0.6.1` (+ `peft`) — both are project
dependencies (`pyproject.toml`, installed by `uv sync`). The upstream fork
checkout **and its release assets** live behind the repo-root `mu0/` symlink
(`point_models/mu0`): `mu0/final_ckpt/` (released checkpoint — `config.json`,
`meta.json`, `model.safetensors`), `mu0/normalizer_stats.json`, the
`mu0/test_set/` droid episodes used by the smoke runs below, and
`mu0/docs/release/TRAINING.md` — the **canonical recipe reference** for the
environment, data layout, stats, and full training command.

Entry points (also installed as `uv run mu0-train` / `uv run mu0-predict`):

- Train / fine-tune — `uv run python -m mu0.scripts.lerobot_train_trace_mu0`
- Eval — `uv run python -m mu0.scripts.lerobot_predict_trace_mu0_image_only`
- CPU port tests — `uv run --extra dev pytest tests/test_mu0_port.py -q`

## Data layout (TraceExtract episodes)

Training and eval read **TraceExtract episode directories** (see the required
files table in TRAINING.md §2). Files the loader actually consumes:

```
<episode>/
  images.npy                   # (T, H, W, 3) uint8 RGB           [required]
  depth.npy                    # float16 metric depth, m          [required when --use_depth=true]
  curated_training_texts.json  # language annotations             [optional]
  samples/
    frame_indices.npy, offsets.npy, is_moving.npy, traj.npy,
    traj_history.npy, valid_steps.npy, valid_steps_history.npy     [required]
    cluster_ids.npy             # per-keypoint DINO cluster id    [required only when rigidity loss is on]
```

Extra arrays the extractor ships (`raw_traj*.npy`, `keypoints.npy`,
`cameras.npz`, …) are ignored — safe to keep or drop. Training episodes are
passed via `--video_dirs`; eval uses `--test_dirs` (see below). Train and eval
use the same sample slots: 8 history + 32 future frames at the release
settings.

## Normalization stats

The model predicts anchor-relative deltas normalized by per-axis scales and
renders depth over a fixed log range; both live in one JSON. Release stats:
`mu0/normalizer_stats.json` (contains `depth_log_min/max` and the
`delta_scale` per-axis `(sx, sy, sz)` — the released ckpt's stored
`delta_scale` matches it). Compute your own with
`uv run python -m mu0.datasets.trace_delta_stats --help` (module form of
TRAINING.md §3's command), or omit `--delta_stats_path` and the trainer
auto-computes stats into the run's output dir on first launch. The stats file
records `target_kind="anchor"`; the trainer validates it and errors on a
mismatch.

## Fine-tuning from the released checkpoint — the primary flow

With `--pretrained_path=<ckpt dir>` the **model architecture is taken from the
checkpoint's own `config.json`**, via the registry-free `from_ckpt_config`
loader (the tokenizer likewise follows the checkpoint's VLM). The flags you
pass shape the **dataset**, and they must match what the checkpoint was
trained on: `--use_depth`, `--history_len`, `--future_len`,
`--trace_group_horizon`, `--trace_bspline_n_ctrl`,
`--trace_bspline_ctrl_per_token`, `--rigidity_regularization_weight`,
`--n_min`/`--n_max`, `--image_size`.

On any CLI/ckpt divergence the trainer warns once at policy build, naming each
drifted field: **model-side fields always win** (CLI values are ignored for the
model — change the architecture only by editing the ckpt's `config.json`), and
the CLI value still shapes the run for the dataset/run side of the compared
fields — so pass them matching the checkpoint. Free to override: `--lr`, the
dropout probabilities (`--trace_history_full_mask_prob`,
`--depth_dropout_prob`, `--trace_history_kp_dropout_prob`), and `delta_scale`
(via the stats file). Everything else divergent is flagged.

Verified smoke command (the dataset-side flags below mirror the release
checkpoint's contract — 2.2B VLM, history 8 / future 32, `n_max` 256, depth
LoRA r8/α16 cloned stem, DINO-base@518, B-spline 10/10, done head):

```bash
# Smoke: 4 finite steps, ~20 s warm / ~4 min cold on an RTX 5090, ~13.6 GB
# peak. A real fine-tune keeps these flags and raises --steps.
uv run python -m mu0.scripts.lerobot_train_trace_mu0 \
  --video_dirs='[mu0/test_set/test_dataset_droid/droid_shard01024_ep010, mu0/test_set/test_dataset_droid/droid_shard01024_ep012]' \
  --delta_stats_path=mu0/normalizer_stats.json \
  --pretrained_path=mu0/final_ckpt \
  --batch_size=1 --num_workers=4 \
  --n_min=1 --n_max=256 \
  --image_size=512 \
  --use_depth=true --rigidity_regularization_weight=5 \
  --history_len=8 --future_len=32 --trace_group_horizon=8 \
  --trace_bspline_n_ctrl=10 --trace_bspline_ctrl_per_token=10 \
  --steps=4 --log_freq=1 --save_freq=3000 --eval_every_n_steps=0 --vis_num_samples=0 \
  --save_checkpoint=false \
  --device=cuda --dtype=bfloat16 \
  --gradient_checkpointing=true \
  --num_inference_steps=4 \
  --lr=1e-4 \
  --wandb_enable=false --wandb_project=mu0 \
  --seed=0 \
  --job_name=smoke_ft
```

Notes:

- The first fine-tune run downloads the `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
  weights (~4.4 GB) if the HF cache is cold (cached on this machine).
- Runs write to `outputs/train/<date>/<time>_<job_name>/` — checkpoints under
  `checkpoints/step_XXXXXXXX/` (`model.safetensors`, `config.json`,
  `meta.json`, `accel_state/`), plus a copy of the resolved `delta_stats.json`.
  Resume with `--resume_from=outputs/train/.../checkpoints/step_XXXXXXXX`
  (TRAINING.md §6).

## Training from scratch

The same trainer without `--pretrained_path`. TRAINING.md §4's command maps
1:1 — replace `python src/lerobot/scripts/lerobot_train_trace_mu0.py` with
`uv run python -m mu0.scripts.lerobot_train_trace_mu0` (or `uv run mu0-train`);
**all flags are identical**. The recipe is sized for one 48 GB GPU
(`--batch_size=2` + gradient checkpointing); `--steps=4
--save_checkpoint=false --eval_every_n_steps=0` turns it into a smoke:

```bash
uv run python -m mu0.scripts.lerobot_train_trace_mu0 \
  --video_dirs='[/path/to/episode_a, /path/to/episode_b]' \
  --delta_stats_path=/path/to/delta_stats.json \
  --batch_size=2 --num_workers=4 \
  --n_min=1 --n_max=256 --image_size=512 \
  --steps=200000 --log_freq=50 --save_freq=3000 \
  --eval_every_n_steps=3000 --eval_episode_num_per_dir=1 --vis_num_samples=5 \
  --device=cuda --dtype=bfloat16 --gradient_checkpointing=true \
  --num_inference_steps=4 \
  --load_vlm_weights=true --vlm_model_name=HuggingFaceTB/SmolVLM2-2.2B-Instruct \
  --num_vlm_layers=20 --num_expert_layers=20 --expert_width_multiplier=0.5 \
  --freeze_vision_encoder=true --train_expert_only=true \
  --lr=1e-4 --wandb_enable=false --wandb_project=mu0 \
  --history_len=8 --future_len=32 --trace_group_horizon=8 \
  --use_depth=true --depth_lora_rank=8 --depth_lora_alpha=16 \
  --use_dino=true --dino_model_name=facebook/dinov2-base --dino_input_size=518 \
  --trace_history_full_mask_prob=0.5 --depth_dropout_prob=0.7 \
  --trace_history_kp_dropout_prob=0.3 --rgb_jitter_strength=0.3 --depth_noise_std=0.01 \
  --trace_bspline_n_ctrl=10 --trace_bspline_ctrl_per_token=10 \
  --trace_done_head=true --done_loss_weight=0.5 \
  --rigidity_regularization_weight=5 \
  --trace_bspline_reg_lambda=0.2 --trace_bspline_reg_order=1 --trace_bspline_ctrl_clip=1.5 \
  --seed=0 \
  --job_name=trace_main
```

## Evaluation (dataset-keypoint mode)

```bash
uv run mu0-predict \
  --checkpoint=mu0/final_ckpt \
  --test_dirs='[mu0/test_set/test_dataset_droid/droid_shard01024_ep010]' \
  --delta_stats_path=mu0/normalizer_stats.json \
  --use_kp_from_dataset=true \
  --use_depth_from=data \
  --output_dir=outputs/mu0_predict/<run_name> \
  --max_samples=8     # bounded smoke; default 0 = whole episode(s)
```

Notes:

- The predict CLI names the episode flag **`--test_dirs`**, NOT `--video_dirs`
  (that is the train CLI's name), and **`--delta_stats_path` is required**
  (verified; a missing stats path fails the run).
- Default `--output_dir` is `<checkpoint>/predictions` — override it (as
  above) so a release ckpt dir is never written into.
- Writes `metrics.json` (plus per-horizon `metrics_8/16.json` from the default
  `--metric_horizons=[8,16,32]`) and overlay PNGs under `rollout_img/`.
  Best-of-`--num_samples_for_metrics` (default 5) min-metrics are reported.
- `--use_kp_from_dataset=true` consumes the dataset's keypoints;
  `--use_depth_from=data|pred|none` (default `pred`). The dataset-keypoint
  modes are the wired, verified paths — fully offline after the first cache.
  **Full image-only synthesis is NOT wired**: `use_kp_from_dataset=false`
  and/or `--use_depth_from=pred` would hit the fork's lazy Grounded-SAM-2 /
  Depth-Anything-V2 imports (`infer_helpers`). It will be re-pointed at this
  repo's `src/depth_models` + `src/det_seg_models` models later.

## Checkpoint compatibility and port fidelity

- The config registration is **`"smolvla_mu0"`, not `"smolvla"`**: 0.6.1's
  vanilla config already holds `"smolvla"` in the shared registry, so μ₀
  checkpoints load **only** through this repo's loader (registry-free
  `from_ckpt_config`). Checkpoints saved by this environment carry
  `"type": "smolvla_mu0"` in their `config.json`; released checkpoints
  (`"type": "smolvla"`) load fine here. Loading one of OUR fine-tunes in the
  upstream fork environment requires a one-line `"type"` edit in its
  `config.json`.
- The port is byte-close to the fork except import paths and three documented
  deviations — the registry-free config loader, the predict `_load_policy`,
  and the fine-tune config derivation (`_derive_ft_smolvla_config` + the
  drift warning) — so future upstream merges diff cleanly.
