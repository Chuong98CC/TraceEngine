# μ₀ trace model port into `src/mu0` — design spec

Date: 2026-09-07 · Status: approved in chat on 2026-09-07 (rev. 2), pending written-spec review

## 1. Goal

Port the **μ₀ trace-prediction model and its training/eval scripts** out of the
research fork `Yoonkyo/mu0` (local checkout `/home/chuong/workspace/point_models/mu0`,
HEAD `1a95186997fb30434d213a6e0b19cb43e8965e11`, itself a LeRobot fork on a
~v0.6.0/0.6.1 base) into a new package **`src/mu0`** in this repo that imports
**only the pip-installed `lerobot==0.6.1`** from the project `.venv` — no
dependency on the fork's patched `lerobot` tree, no edits to vanilla lerobot.

The μ₀-specific code is ~9.4k lines in 13 files. Every `lerobot.*` symbol these
files import has been verified present in the installed 0.6.1 wheel, so the
port is **namespace rewiring, not API surgery** (see §4 import map). Source of
truth for file list: `wc -l` of the fork at HEAD above.

Primary use case going forward: **fine-tune the released μ₀ checkpoint**
(`mu0/final_ckpt` — `config.json` + `model.safetensors` + `meta.json`, trained
to step 204000) on the user's own TraceExtract-format episodes.

## 2. Non-goals (this iteration)

- No `inference_helpers/` port (Grounded-SAM-2 / Depth-Anything-V2 wrappers).
  The repo already has alternatives (`src/depth_models`, `src/det_seg_models`);
  the full image-only eval synthesis path will be re-wired to them later.
- No viser GUI, `lerobot_visualize_trace_3d.py`, GPT/Gemini baseline scripts,
  `lerobot_visualize_bspline_fit.py`.
- No modification of vanilla lerobot: no factory rewiring, no registration into
  lerobot's config registry under `"smolvla"` (0.6.1 ships its own `smolvla`
  policy; the namespaces must not collide).
- No from-scratch pretraining runs in validation (from-scratch path per
  TRAINING.md stays supported by the code, just not exercised here).

## 3. Package layout (mirror of the fork, minus redundant nesting)

The fork's inner `policies/mu0/` nesting is dropped (package root `mu0` already
namespaces it). File names/contents otherwise mirror the fork 1:1.

```
src/mu0/
├── __init__.py                  # port attribution header (source repo + commit) + re-exports
├── datasets/                    # ← fork src/lerobot/datasets/*.py
│   ├── bspline_basis.py         #   120  cubic B-spline basis, precomputed knot vectors
│   ├── trace_depth.py           #    83  metric depth → turbo-colormap RGB (VLM channel)
│   ├── trace_delta_stats.py     #   379  stats CLI — `python -m mu0.datasets.trace_delta_stats`
│   └── trace_dataset.py         #  1076  TraceDataset + trace_collate_fn (TraceExtract episodes)
├── policies/                    # ← fork src/lerobot/policies/mu0/*.py
│   ├── configuration_smolvla.py #   376  SmolVLAConfig(PreTrainedConfig) + trace_* fields
│   ├── smolvlm_with_expert.py   #   726  SmolVLM2 backbone surgery + expert layers + depth LoRA stem
│   ├── modeling_smolvla.py      #  2026  SmolVLAPolicy + VLAFlowMatching (flow-matching, DINO, done head)
│   ├── processor_smolvla.py     #   103  make_smolvla_pre_post_processors
│   └── visualize_trace.py       #   910  trace overlay/motion rendering (cv2/numpy)
└── scripts/
    ├── lerobot_train_trace_mu0.py            # 2250  ← TRAINING.md trainer (draccus CLI)
    └── lerobot_predict_trace_mu0_image_only.py # 1316 ← batch eval (dataset-keypoint modes only)
```

Class names stay **`SmolVLAConfig` / `SmolVLAPolicy`** (subclasses of lerobot
0.6.1's `PreTrainedConfig` / `PreTrainedPolicy`) and the config keeps its
`type: "smolvla"` identity so released `config.json` files round-trip.

## 4. Import rewiring (the only code delta; mechanical)

| In ported files | Rewritten to |
|---|---|
| `lerobot.datasets.{bspline_basis, trace_depth, trace_delta_stats, trace_dataset}` | `mu0.datasets.{…}` (all 4 modules + key constants imported by modeling) |
| `lerobot.policies.mu0.{configuration_smolvla, smolvlm_with_expert, modeling_smolvla, processor_smolvla, visualize_trace}` | `mu0.policies.{…}` |
| `lerobot.scripts.lerobot_train_trace_mu0` (imported by the predict script) | `mu0.scripts.lerobot_train_trace_mu0` |
| relative `..pretrained` / `..rtc.*` / `..utils` (`populate_queues`) inside the ported policy files | absolute `lerobot.policies.pretrained` / `lerobot.policies.rtc.*` / `lerobot.policies.utils` |
| all other `lerobot.*` imports — `utils.constants`, `utils.device_utils`, `utils.import_utils`, `utils.logging_utils`, `utils.utils`, `configs`, `optim`, `processor` | **unchanged** (verified present in 0.6.1 wheel) |

Function-local imports of the fork's `inference_helpers` inside the predict
script stay as-is (code byte-close): they only fire on
`--use_kp_from_dataset=false` / `--use_depth_from=pred`, which are
documented as not-yet-wired. Module docstrings/comments updated only where they
reference old paths; everything else byte-identical for future diffs against
`Yoonkyo/mu0`.

## 5. Repo integration

1. `pyproject.toml`:
   - add `"src/mu0"` to `[tool.hatch.build.targets.wheel] packages`;
   - replace `"lerobot==0.6.1"`, `"lerobot[dataset]"` with
     `"lerobot[dataset,training,smolvla]==0.6.1"` (adds `accelerate`, `wandb`,
     `transformers>=5.4,<5.6`, `num2words`);
   - add `"peft"` (LoRA depth adapters — fork's own uv.lock had to patch this in);
   - add `[project.scripts]`: `mu0-train` → `mu0.scripts.lerobot_train_trace_mu0:main`,
     `mu0-predict` → `mu0.scripts.lerobot_predict_trace_mu0_image_only:main`.
2. `uv sync` (updates `uv.lock`).
3. Commands keep the TRAINING.md shape: `uv run python -m mu0.scripts.lerobot_train_trace_mu0 …`.

## 6. Fine-tune flow (trainer extension — the one functional change)

Fork behavior today: `_build_policy` builds `SmolVLAConfig` **purely from CLI
flags**, then `SmolVLAPolicy.from_pretrained(pretrained_path, config=sv_cfg,
strict=False)`. A CLI-only config can silently diverge from the released
architecture on structural knobs TRAINING.md §4 doesn't surface (verified
release fields: `depth_clone_stem: true`, `trace_attention_mode:
"bidirectional"`, `trace_use_adaln: true`, `causal_attention: true`, …).
Mismatched architecture + `strict=False` = silently re-initialised weights.

Targeted extension, mirroring the fork predict script's own load path
(`SmolVLAConfig` decoded from a checkpoint's `config.json`):

- When `--pretrained_path=<dir>` points at a μ₀ checkpoint whose
  `config.json` has `trace_mode: true`: decode that `config.json` into
  `SmolVLAConfig` and use it as the policy config (architecture parity is
  guaranteed by construction).
- Over it, apply only a **whitelist of runtime-only CLI flags**: learning-rate
  family (`lr`, `weight_decay`, `betas`, `eps`, grad-clip), run structure
  (`batch_size`, `steps`, log/save/eval frequencies), `video_dirs` data flags,
  and `delta_scale`/stats override (from `--delta_stats_path` / auto-computed
  stats — new-domain normalization must win over the checkpoint's stored
  `delta_scale`). Augmentation/dropout probabilities may also override (no
  shape impact). Whitelist membership is enforced in code: any *other* CLI
  flag that differs from the decoded checkpoint config is warned about
  loudly (logged) and ignored.
- `--pretrained_path=None` keeps the fork from-scratch path unchanged.
- `--resume_from` remains mutually exclusive with `--pretrained_path` (already
  enforced in the fork).
- Exact decode mechanics (`draccus.decode(..., target_type=SmolVLAConfig)` vs
  `PreTrainedConfig`-registry helpers) to be copied from the fork predict
  script's `_load_policy` so both scripts share one mechanism.

## 7. Validation (on this machine, 32 GB RTX 5090; weights cached in HF)

1. All 13 ported modules import against pip lerobot 0.6.1 (no fork paths on
   `sys.path`).
2. `mu0/datasets/trace_delta_stats.py` runs as `python -m`; release
   `normalizer_stats.json` loads through `load_trace_stats`.
3. `SmolVLAConfig` decoded from `mu0/final_ckpt/config.json` reproduces the
   released architecture — param-count comparison against
   `model.safetensors` matches (or diffs only by a reported, understood delta).
4. Quick fine-tune per TRAINING.md §4 smoke shape, from `final_ckpt`:
   `--pretrained_path=mu0/final_ckpt --steps=4 --save_checkpoint=false
   --eval_every_n_steps=0`, golden recipe flags, `--batch_size=1`, 2 release
   `test_set` droid episodes, release `normalizer_stats.json`. Expect: clean
   exit, logged losses, no NaNs. (OOM fallback ladder: batch 1 + gradient
   checkpointing already in recipe → last resort swap VLM to the cached
   500M-Video-Instruct default.)
5. Eval path launch: `mu0-predict` (dataset-keypoint + dataset-depth mode)
   against `mu0/final_ckpt` on one droid episode for a bounded frame budget —
   validates the eval side end-to-end without GSAM/DA-v2.
6. From-scratch regression check of the unchanged fork code path: re-run the
   §7.4 command **without `--pretrained_path`** (4 steps, CLI-built config,
   `load_vlm_weights`) so the TRAINING.md §4 default path isn't regressed by
   the §6 extension.

## 8. Docs

- `docs/mu0/training.md` — port notes, run commands for fine-tune and scratch
  training, data-layout pointer to TRAINING.md §2, which eval modes work now
  vs later, source-commit attribution.
- CLAUDE.md package-structure entry for `src/mu0`.

## 9. Risks & mitigations

- **Version drift** (fork env vs ours): torch 2.10→2.11, transformers 5.3→5.4+,
  draccus 0.10→0.11.6, accelerate 1.14. Smoke run (§7.4) is the arbiter; all
  lerobot-level imports already verified present.
- **draccus decode strictness** on `final_ckpt/config.json` (extra/unknown
  keys, `PreTrainedConfig` subclass semantics under 0.6.1) — §7.3 checks it
  first, before any GPU time.
- **Silent architecture drift on fine-tune** — eliminated by §6 design;
  loudly-log guard as backstop.
- **VRAM on 32 GB** — §7 fallback ladder.
- **`test_set` episodes used for a 4-step smoke** — no weights saved
  (`--save_checkpoint=false`), no contamination concern.
- **GPU nondeterminism / OOM in dataloader workers** — resolved at plan/run
  time with the fork's existing knobs (`num_workers`, seeds).

## 10. Source attribution

All ported files derive from `github.com/Yoonkyo/mu0` (Apache-2.0, itself a
LeRobot fork), local mirror HEAD `1a95186997fb30434d213a6e0b19cb43e8965e11`;
the `src/mu0/__init__.py` header records this. See
`mu0/docs/release/TRAINING.md` (training recipe + TraceExtract episode layout)
and `mu0/docs/release/EVALUATION.md` (eval modes) in that checkout.
