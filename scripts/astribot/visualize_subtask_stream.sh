# Case 2, Step 2 — visualize: render one trajectory video per sub-task segment
# from the saved depth_pose outputs (colour frames decoded online from the
# dataset; no re-inference). Camera/selection flags must match the Step-2 run.
DATA_ROOT=/data/astri_making_coffee_v1
REPO_ID=Kronze157/astri_making_coffee_vlva

python tools/astribot/visualize_subtask_stream.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --camera-idxes 4 5 --fps 30 \
    --use-inferred-splits
