# Case 2, Step 4 — visualize: render per-camera trace videos (trace2d.mp4 +
# trace3d.mp4) of every sub-task under
# <data-root>/eps_data/visualization/<ep>/<subtask_XX>/<camera>/, from the
# saved run_step4_traces.py outputs (RGB frames decoded online from the
# dataset, no re-inference). Episodes/sub-tasks/cameras are discovered
# from the saved Step-4 trace outputs (Step-2 geometry from the
# depth_pose tree); -e/-c merely filter what exists on disk.
REPO_ID=Kronze157/astri_making_coffee_vlva
DATA_ROOT=/data/astri_making_coffee_v1

python tools/astribot/visualize_step4_traces.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --camera-idxes 0 4 5  --render both
