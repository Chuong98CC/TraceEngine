#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX checkpoint inside the NVIDIA TensorRT container.
#
# Usage:
#   ./scripts/general_test/export_trt_docker.sh <onnx_ckpt_path> [precision]
#
# Arguments:
#   onnx_ckpt_path    Absolute path to the ONNX file on the host,
#                     e.g. /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx
#   precision         Optional TRT precision: tf32 (default) or fp32
#
# Example:
#   ./scripts/export_trt_docker.sh \
#       /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx tf32
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <onnx_ckpt_path> [precision]" >&2
    exit 1
fi

ONNX_CKPT_PATH="$1"
PRECISION="${2:-tf32}"

if [ ! -f "$ONNX_CKPT_PATH" ]; then
    echo "ONNX file not found: $ONNX_CKPT_PATH" >&2
    exit 1
fi

# Host tools dir is the repo-root tools/ directory (this script sits in
# scripts/general_test/, two levels below the repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_TOOLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/tools"

HOST_WEIGHTS_DIR="$(cd "$(dirname "$ONNX_CKPT_PATH")" && pwd)"
ONNX_NAME="$(basename "$ONNX_CKPT_PATH")"

# install tensorrt runtime in your env with: pip install tensorrt-cu12==11.1.0.106
DOCKER_IMG="nvcr.io/nvidia/tensorrt:26.07-py3"
DOCKER_WORKDIR="/workspace"

ONNX_PATH="/weights/$ONNX_NAME"
TRT_PATH="/weights/${ONNX_NAME%.onnx}_${PRECISION}.engine"

docker run --gpus all -it --rm \
    -v "$HOST_WEIGHTS_DIR":/weights \
    -v "$HOST_TOOLS_DIR":$DOCKER_WORKDIR \
    -w $DOCKER_WORKDIR $DOCKER_IMG \
    bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision $PRECISION"
