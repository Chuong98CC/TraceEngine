LEFT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_left"
RIGHT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_right"
LEFT_MASK_DIR="demo_data/astribot_stereo_lrb/motion_mask/stereo_left/mask"
RIGHT_MASK_DIR="demo_data/astribot_stereo_lrb/motion_mask/stereo_right/mask"
# BACKEND="da3"
BACKEND="vggt_omega"
MODEL_PATH="weights/vggt_omg/vggt_omg_64x640x480_bf16.pt2"

python tools/general_test/run_stream.py \
    --backend $BACKEND \
    --input-dirs $LEFT_DIR $RIGHT_DIR \
    --start-frame 210 --max-frames 160 --interval 4 \
    --model-path $MODEL_PATH \
    --output-dir output/stream_stereo_${BACKEND}_cs64_mask \
    --mask-dirs $LEFT_MASK_DIR $RIGHT_MASK_DIR
