REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/run_step3_init_points.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --camera-idxes 0 4 5 \
    --use-inferred-splits \
    --object-top-k 64 --manipulator-top-k 128 \
    --sampling-mode no_roma \
    --with-optical-flow-mask --visualize-motion --motion-ratio 0.03

