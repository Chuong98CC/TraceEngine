#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX checkpoint inside the NVIDIA TensorRT container.
#
# Usage:
#   ./scripts/export_trt_docker.sh <onnx_ckpt_path> [precision]
#
# Arguments:
#   onnx_ckpt_path    Absolute path to the ONNX file on the host,
#                     e.g. /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx
#   precision         Optional TRT precision: fp16 (default), tf32, or fp32
#
# Example:
#   ./scripts/export_trt_docker.sh \
#       /home/chuong/workspace/depth_models/Depth-Anything-3/weights/da3_metric_644x490_large.onnx fp16
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

# Host tools dir is the tools/ directory (sibling to this script's scripts/ directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_TOOLS_DIR="$(dirname "$SCRIPT_DIR")/tools"

# TensorRT 11 removed the --fp16 trtexec flag: fp16 engines need a strongly-typed
# fp16 ONNX.  The tensorrt container ships no onnx tooling, so cast host-side
# (uv env) and build from the casted file.
ORIG_ONNX_NAME="$(basename "$ONNX_CKPT_PATH")"
if [ "$PRECISION" = "fp16" ] && [[ "$ORIG_ONNX_NAME" != *_fp16.onnx ]]; then
    CASTED_ONNX="${ONNX_CKPT_PATH%.onnx}_fp16.onnx"
    if [ ! -f "$CASTED_ONNX" ]; then
        echo "Casting to fp16: $CASTED_ONNX"
        (cd "$(dirname "$SCRIPT_DIR")" && uv run python tools/cast_onnx_fp16.py "$ONNX_CKPT_PATH" "$CASTED_ONNX")
    fi
    ONNX_CKPT_PATH="$CASTED_ONNX"
fi

HOST_WEIGHTS_DIR="$(cd "$(dirname "$ONNX_CKPT_PATH")" && pwd)"
ONNX_NAME="$(basename "$ONNX_CKPT_PATH")"

# install tensorrt runtime in your env with: pip install tensorrt-cu12==11.1.0.106
DOCKER_IMG="nvcr.io/nvidia/tensorrt:26.07-py3"
DOCKER_WORKDIR="/workspace"

ONNX_PATH="/weights/$ONNX_NAME"
# Name the engine after the ORIGINAL onnx stem (not the casted one).
TRT_PATH="/weights/${ORIG_ONNX_NAME%.onnx}_${PRECISION}.engine"

docker run --gpus all -it --rm \
    -v "$HOST_WEIGHTS_DIR":/weights \
    -v "$HOST_TOOLS_DIR":$DOCKER_WORKDIR \
    -w $DOCKER_WORKDIR $DOCKER_IMG \
    bash -c "python export_trt.py $ONNX_PATH --trt_path $TRT_PATH --precision $PRECISION"
