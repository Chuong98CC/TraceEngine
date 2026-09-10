# Case 2, Step 1 — infer the sub-task split frames from the gripper state.
# detect_subtask needs no video decode: it writes subtask_splits.json + the
# gripper plot under eps_data/subtask/<episode>/. Skip this step when the
# dataset's ground-truth subtask_index column is used instead (drop
# --use-inferred-splits from the Step-2 commands).
DATA_ROOT=/data/astribot_making_coffee_vlva_full
REPO_ID=Kronze157/astribot_making_coffee_vlva_full
MODE=${1:-frames}  # frames | detect_subtask
echo "Please select the mode from the following options: detect_subtask| key_frames | frames | videos"
echo "Extracting frames for $REPO_ID from $DATA_ROOT in mode $MODE"

python tools/astribot/extract_frames.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --use-inferred-splits \
    --mode $MODE --interval 4 -c 0 \
    # --mode detect_subtask
