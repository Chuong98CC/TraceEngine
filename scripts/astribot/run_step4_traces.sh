# Case 2, Step 4 — per-sub-task 3D point tracking with TAPIP3D, online (RGB
# frames decoded from the dataset one window at a time, geometry from the
# Step-2 depth_pose outputs, init points from the Step-3 init_points
# outputs — nothing extracted to disk). One pass per role: the object
# keypoints are anchored at the sub-task's first usable key-frame, the
# manipulator keypoints at the sub-task's first frame. Results land under
# eps_data/traces/. Requires Step 2 (depth_pose) and Step 3 (init_points)
# for the same camera (the one Step 3 recorded — run run_step2_depth_stream.py
# without --camera-idxes to stream that camera).
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/run_step4_traces.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --out-dir $DATA_ROOT/eps_data
