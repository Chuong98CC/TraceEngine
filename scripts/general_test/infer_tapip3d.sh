#!/usr/bin/env bash
# TAPIP3D streaming long-video inference on the ONNX models (16-frame windows,
# Tapip3D_ONNX + StreamONNX). Mirror of infer_stream_stereo.sh: image size and query
# count are auto-detected from the ONNX graphs. The shipped updater was
# exported for N = 1088 queries: 8x8 bbox grid (64) + 32x32 support (1024),
# hence --grid_x 8 --grid_y 8 (the PyTorch script uses 4x4).
# Torso: --bbox 70 357 133 396
# Left stereo: --bbox 200 265 260 300
set -e

IMG_DIR="${IMG_DIR:-demo_data/astribot_stereo_lrb/extract_frames/stereo_left}"
GEO_DIR="${GEO_DIR:-demo_data/astribot_stereo_lrb/geometry/stereo_left}"
OUTPUT_DIR="${OUTPUT_DIR:-output/tapip3d_stereo_left_onnx}"
ENCODER="${ENCODER:-weights/tapip3d/tapip3d_encoder_480x640.onnx}"
UPDATER="${UPDATER:-weights/tapip3d/tapip3d_updater.onnx}"
CORR_FORWARD="${CORR_FORWARD:-weights/tapip3d/tapip3d_corr_forward.onnx}"

echo "============================================"
echo "TAPIP3D Streaming ONNX Inference"
echo "============================================"
echo "Image dir:   $IMG_DIR"
echo "Geo dir:     $GEO_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "Encoder:     $ENCODER"
echo "Updater:     $UPDATER"
echo "Corr forward: $CORR_FORWARD"
echo "============================================"

python tools/general_test/infer_tapip3d.py \
    --encoder "$ENCODER" \
    --updater "$UPDATER" \
    --corr_forward "$CORR_FORWARD" \
    --image_dir "$IMG_DIR" \
    --npz_dir "$GEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_frame 210 \
    --fps 4 \
    --bbox 200 265 260 300 \
    --grid_x 8 \
    --grid_y 8 \
    --support_grid_size 32 \
    --num_iters 6 \
    --vis_threshold 0.5 --visualize
    # --max_frames 180 \

echo "============================================"
echo "Done! Results saved to $OUTPUT_DIR"
echo "============================================"
