# VID=demo_data/astribot_stereo_lrb/videos/observation.images.cam_head_stereo_left/chunk-000/file_convertx264.mp4
IMG_DIR=demo_data/astribot_stereo_lrb/extract_frames/stereo_right
TRT_ENGINE=weights/waftv2/waftv2_dinov3_i5_640x480_tf32.engine
OUT_DIR=demo_data/astribot_stereo_lrb/motion_mask/
python tools/general_test/infer_waft.py --input "$IMG_DIR" --backend trt \
    --checkpoint "$TRT_ENGINE" \
    --start 210 --stride 4 -thr 2 -o mask \
    --output-dir "$OUT_DIR"
