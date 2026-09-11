# Case 2, Step 4 — visualize: render per-camera trace videos (trace2d.mp4 +
# trace3d.mp4) of every sub-task under
# <data-root>/eps_data/visualization/<ep>/<subtask_XX>/<camera>/, from the
# saved run_step4_traces.py outputs (RGB frames decoded online from the
# dataset, no re-inference). Episodes/sub-tasks/cameras are discovered
# from the saved Step-4 trace outputs (Step-2 geometry from the
# depth_pose tree); -e/-c merely filter what exists on disk.
REPO_ID=Kronze157/astribot_making_coffee_vlva_full
DATA_ROOT=/data/astribot_making_coffee_vlva_full
EPS=${1:-0}  # episode index

python tools/astribot/clear_task.py \
    --data-root $DATA_ROOT --task visualization \
    --episode-idxes $EPS

python tools/astribot/visualize_step4_traces.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes $EPS \
    --camera-idxes 0 --render 2d
    # --render stills
