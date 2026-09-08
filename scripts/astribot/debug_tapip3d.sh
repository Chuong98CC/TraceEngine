# Case 2, Step 4 — debug: track ONE Step-3 prompt's init points with the
# TAPIP3D .pt2 programs, in the infer_tapip3d.py style (online frames from
# the dataset, Step-2 depth_pose geometry, 2D tracks.mp4 rendered via the
# known-good render_tracks visualizer).
#
# Currently pointed at the case with wrong output: episode 0, subtask_00,
# cam_head_stereo_left (dataset camera idx 4), the manipulator prompt.
IMG_DIR="cache/step4_debug/frames"
DEPTH_DIR="astri_making_coffee_v1/eps_data/depth_pose/ep000000/subtask_00/depth_cam_head_stereo_left"
OUTPUT_DIR="cache/debug_tapip3d_infer"


echo "============================================"
echo "TAPIP3D Streaming PT2 Inference"
echo "============================================"
echo "Image dir:   $IMG_DIR"
echo "Depth dir:   $DEPTH_DIR"
echo "Output dir:  $OUTPUT_DIR"
echo "============================================"

python tools/general_test/module/infer_tapip3d.py \
    --image_dir "$IMG_DIR" \
    --depth_dir "$DEPTH_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --start_frame 0 \
    --interval 1 \
    --bbox 147 258 241 318  --text_prompt "left robot arm's black grippers" \
    --grid_x 8 \
    --grid_y 8 \
    --support_grid_size 32 \
    --num_iters 6 \
    --vis_threshold 0.5 --visualize
