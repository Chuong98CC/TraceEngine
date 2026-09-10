# Case 2, Step 3 — key-point sampling: extract the episode's key-frame jpgs
# (Step 1), then RexOmni detections (3a) + SAM3/RoMaV2 init points (3b).
# Add --with-optical-flow-mask to also run Step 3a' (WAFT motion masks,
# motion_masks/) and union them into the manipulator's SAM mask in 3b; add
# --visualize-motion to also write each chosen pair's flow visuals.
# Outputs land (by default) under eps_data/sampling_points/key_frames/,
# .../sampling_points/detections/ and .../sampling_points/init_points/.
# Run from the main env (Step 3a uses the separate .venv-rexomni env
# internally).
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

WITH_OPTICAL_FLOW_MASK=${1:-false}  # true: run Step 3a' (WAFT motion masks) and union them into the manipulator's SAM mask in 3b
if [ "$WITH_OPTICAL_FLOW_MASK" = true ]; then
    echo "Running Step 3a' (WAFT motion masks) and union them into the manipulator's SAM mask in 3b"
    WITH_OPTICAL_FLOW_MASK_FLAG="--with-optical-flow-mask"
    python tools/astribot/run_step3_motion_masks.py \
        --repo-id $REPO_ID \
        --data-root $DATA_ROOT --episode-idxes 0 \
        --camera-idxes 0 4 5 \
        --stride 4 --motion-ratio 0.05 --visualize
else
    echo "Skipping Step 3a' (WAFT motion masks)"
    WITH_OPTICAL_FLOW_MASK_FLAG=""
fi

# python tools/astribot/run_step3_init_points.py \
#     --repo-id $REPO_ID \
#     --data-root $DATA_ROOT --episode-idxes 0 \
#     --camera-idxes 0 4 5 \
#     --use-inferred-splits \
#     --object-top-k 64 --manipulator-top-k 128 \
#     --sampling-mode no_roma \  # other modes: uniform, mask, no_roma
#     $WITH_OPTICAL_FLOW_MASK_FLAG
