#!/usr/bin/env bash
# TAPIP3D streaming long-video inference on the ONNX models (16-frame windows,
# Tapip3D_ONNX + StreamONNX). Mirror of infer_stream_stereo.sh: image size and query
# count are auto-detected from the ONNX graphs. The shipped updater was
# exported for N = 1088 queries: 8x8 bbox grid (64) + 32x32 support (1024),
# hence --grid_x 8 --grid_y 8 (the PyTorch script uses 4x4).
# Torso: --bbox 70 357 133 396
# Left stereo: --bbox 200 265 260 300
set -e

IMG_DIR="${IMG_DIR:-/data/astri_making_coffee_v1/eps_data/subtask_frames/ep000000/subtask_00/cam_head}"
DEPTH_DIR="${DEPTH_DIR:-/data/astri_making_coffee_v1/experiments/rgbd_a2f/depth_cam_head}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/astri_making_coffee_v1/experiments/tapip3d}"


echo "============================================"
echo "TAPIP3D Streaming PT2 Inference"
echo "============================================"
echo "Image dir:   $IMG_DIR"
echo "Depth dir:   $DEPTH_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "============================================"

python tools/general_test/infer_tapip3d.py \
    --image_dir "$IMG_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_frame 0 \
    --interval 1 \
    --bbox 1 240 100 340 --text_prompt "brown coffee cup" \
    --grid_x 8 \
    --grid_y 8 \
    --support_grid_size 32 \
    --num_iters 6 \
    --vis_threshold 0.5 --visualize
    # --max_frames 180 \

echo "============================================"
echo "Done! Results saved to $OUTPUT_DIR"
echo "============================================"
