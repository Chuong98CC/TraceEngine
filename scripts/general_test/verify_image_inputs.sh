#!/usr/bin/env bash
# Golden capture/compare for the image-input refactor (spec 2026-09-02).
# Usage:
#   verify_image_inputs.sh [capture|compare] [da3|vggt|romav2|sam3|any2full|tapip3d|all] [extra args...]
# The runtime slot may be skipped; every remaining arg (from the first flag on)
# is forwarded to compare_image_inputs.py, e.g.
#   verify_image_inputs.sh compare --frames-dir <dir>
#   verify_image_inputs.sh compare da3 --frames-dir <dir>
# The default frames dir (assets/astribot_test_imgs/head_rgbd) holds a single
# jpg, so capture/compare need --frames-dir pointing at a folder with enough
# RGB jpgs (>= num_views=64 for the e2e probes), e.g. an extracted-frames folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:-capture}"
RUNTIME="all"
FORWARD=()
if [[ $# -ge 2 ]]; then
    if [[ "$2" != -* ]]; then
        RUNTIME="$2"                 # runtime given as the 2nd positional
        FORWARD=("${@:3}")
    else
        FORWARD=("${@:2}")           # runtime slot skipped: forward as-is
    fi
fi
uv run python tools/general_test/module/compare_image_inputs.py \
    --mode "$MODE" --runtime "$RUNTIME" "${FORWARD[@]}"
