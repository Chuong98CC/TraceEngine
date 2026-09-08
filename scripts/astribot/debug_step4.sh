# Case 2, Step 4 — debug: track ONE Step-3 prompt's init points with the
# TAPIP3D .pt2 programs, in the infer_tapip3d.py style (online frames from
# the dataset, Step-2 depth_pose geometry, 2D tracks.mp4 rendered via the
# known-good render_tracks visualizer).
#
# Currently pointed at the case with wrong output: episode 0, subtask_00,
# cam_head_stereo_left (dataset camera idx 4), the manipulator prompt.
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1
EPISODE_IDX=0
SUBTASK_IDX=00
CAMERA_IDX=4
PROMPT=left_robot_arm_s_black_grippers
OUT_DIR=cache/step4_debug

python tools/astribot/debug_track_step3_points.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idx $EPISODE_IDX \
    --subtask-idx $SUBTASK_IDX \
    --camera-idx $CAMERA_IDX \
    --prompt $PROMPT \
    --out-dir $OUT_DIR
