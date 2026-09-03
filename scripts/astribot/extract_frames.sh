# Case 2, Step 1 — infer the sub-task split frames from the gripper state.
# detect_subtask needs no video decode: it writes subtask_splits.json + the
# gripper plot under eps_data/subtask/<episode>/. Skip this step when the
# dataset's ground-truth subtask_index column is used instead (drop
# --use-inferred-splits from the Step-2 commands).
DATA_ROOT=/data/astri_making_coffee_v1
REPO_ID=Kronze157/astri_making_coffee_vlva

python tools/astribot/extract_frames.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --mode detect_subtask
