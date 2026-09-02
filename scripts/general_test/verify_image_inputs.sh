#!/usr/bin/env bash
# Golden capture/compare for the image-input refactor (spec 2026-09-02).
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-capture}"   # capture | compare
RUNTIME="${2:-all}"    # da3|vggt|romav2|sam3|any2full|tapip3d|all
uv run python tools/general_test/module/compare_image_inputs.py \
    --mode "$MODE" --runtime "$RUNTIME"
