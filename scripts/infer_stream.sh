LEFT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_left"
RIGHT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_right"
LEFT_MASK_DIR="demo_data/astribot_stereo_lrb/motion_mask/stereo_left/mask"
RIGHT_MASK_DIR="demo_data/astribot_stereo_lrb/motion_mask/stereo_right/mask"
BACKEND="da3"
# BACKEND="vggt_omega"
ANYVIEW_MODEL_PATH="weights/da3/da3_anyview_64x644x490_giant-large-1.1.pt2"
METRIC_MODEL_PATH="weights/da3/da3_metric_644x490_giant-large-1.1.pt2"

python tools/general_test/run_stream.py \
    --backend $BACKEND \
    --input-dirs $LEFT_DIR $RIGHT_DIR \
    --start-frame 210 --max-frames 160 --interval 4 \
    --anyview-model-path $ANYVIEW_MODEL_PATH \
    --metric-model-path $METRIC_MODEL_PATH \
    --output-dir output/da3_${BACKEND}_cs64_mask \
    --mask-dirs $LEFT_MASK_DIR $RIGHT_MASK_DIR
