# IMG="assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg"
# IMG="assets/astribot_test_imgs/torso_rgbd/color/img_000000.jpg"
IMG="assets/astribot_test_imgs/head_rgbd/color/img_000000.jpg"
PYTHONPATH="$PWD/src" .venv-rexomni/bin/python tools/general_test/module/infer_rexomni.py \
      -i "$IMG" \
    --prompts "brown cup" "left black robot grippers"  "right black robot grippers" "coffee maker's control panel" "drip tray of coffee maker" \
    --out_dir cache/rexomni
