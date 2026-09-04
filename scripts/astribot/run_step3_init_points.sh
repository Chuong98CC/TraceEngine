# Case 2, Step 3 — key-point sampling: extract the episode's key-frame jpgs
# (Step 1), then RexOmni detections (3a) + SAM3/RoMaV2 init points (3b).
# Outputs land under eps_data/key_frames/, eps_data/detections/ and
# eps_data/init_points/. Run from the main env (Step 3a uses the separate
# .venv-rexomni env internally).
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/run_step3_init_points.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --out-dir $DATA_ROOT/eps_data \
    --use-inferred-splits
