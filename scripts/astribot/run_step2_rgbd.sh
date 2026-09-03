# Case 2, Step 2 — RGB-D variant: Any2Full (--backend a2f) densifies the raw
# sensor depth (uint16 mm) of a camera with a paired depth feature. Online,
# nothing extracted to disk; requires the Step-1 subtask_splits.json when
# --use-inferred-splits is passed.
DATA_ROOT=/data/astri_making_coffee_v1
REPO_ID=Kronze157/astri_making_coffee_vlva
# BACKEND=da3
BACKEND=a2f
CAMERA_IDX=3   # cam_torso (a camera with a paired raw-depth feature)

# --no-depth-enhance: skip Any2Full and feed the raw sensor depth into the
# alignment step
python tools/astribot/run_step2_depth_stream.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --camera-idxes $CAMERA_IDX --backend $BACKEND \
    --with-optical-flow \
    --use-inferred-splits

# render the trajectory videos of this camera:
# bash scripts/astribot/visualize_subtask_stream.sh -c $CAMERA_IDX \
#     --fps 15 --size 960x540
