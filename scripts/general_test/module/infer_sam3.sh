# IMG="assets/astribot_test_imgs/head_stereo_left/frame_000210.jpg"
# IMG="assets/astribot_test_imgs/torso_rgbd/color/img_000000.jpg"
IMG="assets/astribot_test_imgs/head_rgbd/color/img_000000.jpg"

python tools/general_test/module/infer_sam3.py \
      -i $IMG  --conf 0.5    \
      --prompts "brown cup" "left robot gripper" "right robot arm" "coffee machine" \
      --out_dir cache/sam3