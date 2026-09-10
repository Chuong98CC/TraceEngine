# Case 2, Step 2 — visualize: render one depth_pose.mp4 per (sub-task,
# camera) from the saved depth_pose outputs, under
# <data-root>/eps_data/visualization/<ep>/<subtask_XX>/<camera>/ (colour
# frames decoded online from the dataset; no re-inference). Episodes/
# segments/cameras are discovered from the saved Step-2 outputs; -e/-c
# merely filter what exists on disk.
DATA_ROOT=/data/astribot_making_coffee_vlva_full
REPO_ID=Kronze157/astribot_making_coffee_vlva_full

python tools/astribot/visualize_step2_depth_pose.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --episode-idxes 0 \
    --camera-idxes 0  --fps 30
