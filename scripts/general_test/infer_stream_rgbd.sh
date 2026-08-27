DATA_ROOT="/data/astri_making_coffee_v1/"
RGB_DIR="${DATA_ROOT}/eps_data/subtask_frames/ep000000/subtask_00/cam_head"
DEPTH_DIR="${DATA_ROOT}/eps_data/subtask_frames/ep000000/subtask_00/depth_cam_head"
# BACKEND="da3"
BACKEND="a2f"
OUTPUT_DIR="${DATA_ROOT}/experiments/rgbd_${BACKEND}"

# run inference on the RGB-D stream
# python tools/general_test/run_stream.py \
#     --backend $BACKEND \
#     --input-dirs $RGB_DIR \
#     --depth-dirs $DEPTH_DIR \
#     --start-frame 0 --max-frames 160 --interval 1 \
#     --output-dir $OUTPUT_DIR
#     # --mask-dirs $LEFT_MASK_DIR $RIGHT_MASK_DIR

# visualize
python tools/general_test/visualize_stream.py \
    --input-dirs $RGB_DIR \
    --result-dir $OUTPUT_DIR --output $OUTPUT_DIR/rgbd_${BACKEND}_stream.mp4 \
    --fps 15 --size 960x540
