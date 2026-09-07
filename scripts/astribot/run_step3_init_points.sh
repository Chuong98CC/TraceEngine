# Case 2, Step 3 — key-point sampling: extract the episode's key-frame jpgs
# (Step 1), then RexOmni detections (3a) + SAM3/RoMaV2 init points (3b).
# Outputs land (by default) under eps_data/sampling_points/key_frames/,
# .../sampling_points/detections/ and .../sampling_points/init_points/.
# Run from the main env (Step 3a uses the separate .venv-rexomni env
# internally).
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/run_step3_init_points.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --use-inferred-splits \
    --camera-idx 0 3 4 5
