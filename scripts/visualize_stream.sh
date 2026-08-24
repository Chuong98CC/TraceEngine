LEFT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_left"
RIGHT_DIR="demo_data/astribot_stereo_lrb/extract_frames/stereo_right"
OUTPUT_DIR="output/stream_stereo_vggt_omega_cs64_mask"

# python da3_streaming/visualize_glb.py --stem frame_000211 \
#     --input-dirs $LEFT_DIR $RIGHT_DIR \
#     --result-dir $OUTPUT_DIR --output $OUTPUT_DIR/frame_000211.glb

python tools/general_test/visualize_stream.py \
    --input-dirs $LEFT_DIR $RIGHT_DIR \
    --result-dir $OUTPUT_DIR --output $OUTPUT_DIR/vggt_omega_stream.mp4 \
    --fps 30 --size 960x540