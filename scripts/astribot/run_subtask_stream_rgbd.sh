DATA_ROOT="/data/astri_making_coffee_v1/"
REPO_ID="Kronze157/astri_making_coffee_vlva"
# BACKEND="da3"
BACKEND="a2f"
CAMERA_IDX=3   # cam_torso (a camera with a paired raw-depth feature)

# run inference on the RGB-D stream (online, straight from the dataset;
# a2f densifies the sensor depth with Any2Full)
# python tools/astribot/run_subtask_stream.py \
#     --repo-id $REPO_ID \
#     --data-root $DATA_ROOT \
#     --episode-idxes 0 \
#     --camera-idxes $CAMERA_IDX --backend $BACKEND
#     # --no-depth-enhance
    # --with-optical-flow

# visualize
python tools/astribot/visualize_subtask_stream.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT --episode-idxes 0 \
    --camera-idxes $CAMERA_IDX --fps 15 --size 960x540
