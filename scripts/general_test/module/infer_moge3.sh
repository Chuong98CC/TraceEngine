# IMG_DIR=/data/astribot_making_coffee_vlva_full/eps_data/sampling_points/key_frames/ep000000/subtask_00/cam_head
IMG_DIR='cache/sample_images/robotwin'
OUT_DIR='cache/moge/robotwin'
python tools/general_test/module/infer_moge3.py \
    --input $IMG_DIR \
    --out_dir $OUT_DIR  \
    --visualize --save-packed --save-blend-Dab \
    --refine_steps 3
    # --save-Alb-norm --save-L-norm --save-fused \
