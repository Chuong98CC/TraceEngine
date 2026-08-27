DATA_ROOT="/data/astri_making_coffee_v1/"
LEFT_DIR="${DATA_ROOT}/eps_data/subtask_frames/ep000000/subtask_00/cam_head_stereo_left"
RIGHT_DIR="${DATA_ROOT}/eps_data/subtask_frames/ep000000/subtask_00/cam_head_stereo_right"
# BACKEND="da3"
BACKEND="vggt_omega"
OUTPUT_DIR="${DATA_ROOT}/experiments/stereo_${BACKEND}"
ANYVIEW_MODEL_PATH="weights/da3/da3_anyview_64x644x490_giant-large-1.1.pt2"
METRIC_MODEL_PATH="weights/da3/da3_metric_644x490_giant-large-1.1.pt2"

# run inference on stereo stream
# python tools/general_test/run_stream.py \
#     --backend $BACKEND \
#     --input-dirs $LEFT_DIR $RIGHT_DIR \
#     --start-frame 0 --max-frames 160 --interval 1 \
#     --anyview-model-path $ANYVIEW_MODEL_PATH \
#     --metric-model-path $METRIC_MODEL_PATH \
#     --output-dir $OUTPUT_DIR \
#     # --mask-dirs $LEFT_MASK_DIR $RIGHT_MASK_DIR

# visualize
python tools/general_test/visualize_stream.py \
    --input-dirs $LEFT_DIR $RIGHT_DIR \
    --result-dir $OUTPUT_DIR --output $OUTPUT_DIR/stereo_${BACKEND}_stream.mp4 \
    --fps 15 --size 960x540