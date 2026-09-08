# Case 2, Step 4 — visualize: render per-camera trace videos (traces_2d.mp4
# + traces_3d.mp4) of every sub-task from the saved run_step4_traces.py
# outputs (RGB frames decoded online from the dataset, no re-inference).
# Selection flags must match the Step-4 run.
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/visualize_subtask_traces.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0
