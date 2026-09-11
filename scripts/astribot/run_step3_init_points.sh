# Case 2, Step 3 — key-point sampling: extract the episode's key-frame jpgs
# (Step 1), then RexOmni detections (3a) + SAM3/RoMaV2 init points (3b).
# Outputs land (by default) under eps_data/sampling_points/key_frames/,
# .../sampling_points/detections/ and .../sampling_points/init_points/.
# Run from the main env (Step 3a uses the separate .venv-rexomni env
# internally).
REPO_ID=Kronze157/astribot_making_coffee_vlva_full
DATA_ROOT=/data/astribot_making_coffee_vlva_full

# sampling modes: uniform, mask, no_roma
# (no_roma weights the manipulator's draw toward the manipulated object's
#  detection-box center; --no-manipulator-near-object makes it uniform)
python tools/astribot/run_step3_init_points.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --camera-idxes 0  \
    --use-inferred-splits \
    --object-top-k 64 --manipulator-top-k 128 \
    --sampling-mode no_roma \
    --with-optical-flow-mask --visualize --motion-ratio 0.03

