# Case 2, Step 2 — per-sub-task depth + pose streaming, online (frames decoded
# from the dataset one chunk at a time, nothing extracted to disk). Requires
# the Step-1 subtask_splits.json when --use-inferred-splits is passed.
# Stereo: head cameras 4+5 share every 64-view chunk (32 steps x 2 cameras);
# WAFT motion masks zero the moving pixels' confidence during chunk alignment.
DATA_ROOT=/data/astri_making_coffee_v1
REPO_ID=Kronze157/astri_making_coffee_vlva
# BACKEND=da3
BACKEND=vggt_omega

python tools/astribot/run_step2_depth_stream.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --camera-idxes 4 5 --backend $BACKEND \
    --with-optical-flow \
    --use-inferred-splits

# render the trajectory videos: bash scripts/astribot/visualize_subtask_stream.sh
