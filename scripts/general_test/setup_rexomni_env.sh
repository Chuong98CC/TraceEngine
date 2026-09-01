#!/usr/bin/env bash
set -euo pipefail

# Rex-Omni's tested stack is intentionally isolated from the main Python 3.12
# environment (the main project uses torch 2.11 / transformers 5.x).
ENV_DIR="${REXOMNI_ENV_DIR:-.venv-rexomni}"
PYTHON_VERSION="${REXOMNI_PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

uv venv "$ENV_DIR" --python "$PYTHON_VERSION"
uv pip install --python "$ENV_DIR/bin/python" \
  torch==2.7.0 torchvision \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ENV_DIR/bin/python" \
  transformers==4.51.3 \
  accelerate==1.10.1 \
  qwen_vl_utils==0.0.14 \
  'flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4+cu128torch2.7-cp310-cp310-linux_x86_64.whl'
# Step 3a (tools/general_test/pipeline/run_object_detection.py) runs on the
# key-frames that Step 1 saved to disk (extract_frames.py --mode key_frames),
# so only tqdm is needed on top of the model stack (PIL comes with
# transformers; no lerobot — it requires Python >= 3.12)
uv pip install --python "$ENV_DIR/bin/python" tqdm

echo
printf 'Rex-Omni environment ready: %s/bin/python\n' "$ENV_DIR"
printf 'Run: PYTHONPATH=%s/src %s/bin/python %s/tools/general_test/module/infer_rexomni.py\n' \
  "$REPO_ROOT" "$ENV_DIR" "$REPO_ROOT"
