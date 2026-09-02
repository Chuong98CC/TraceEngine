# Astribot camera frame
python tools/general_test/visualize_rgbd.py --camera_name head_rgbd \
    --frame_index 0 --save_viz --save_glb

# RGB + depth-npz folder pair (e.g. a run_stream.py output)
python tools/general_test/pipeline/visualize_rgbd.py \
    --rgb_dir assets/astribot_test_imgs/head_stereo_left \
    --depth_npz_dir assets/astribot_test_imgs/head_stereo_left \
    --save_viz --save_glb
