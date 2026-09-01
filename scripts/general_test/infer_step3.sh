# #!/usr/bin/env bash
# set -euo pipefail
#
# # Step 3 on a folder of key-frame images (one sub-task of one camera):
# # RexOmni detections (3a) + SAM3/RoMAv2 init points (3b), run in sequence.
# # Dataset-independent: the folder is the only input. Run from the main env.
# KEYFRAMES_DIR="${1:?usage: infer_step3.sh <key-frames-dir> [extra args...]}"
# shift
# python tools/general_test/pipeline/run_e2e_init_points.py \
#     --keyframes-dir "$KEYFRAMES_DIR" "$@"

IMG_DIR='astri_making_coffee_v1/eps_data/key_frames/ep000000/subtask_00/cam_head'
python tools/general_test/pipeline/run_e2e_init_points.py --keyframes-dir $IMG_DIR \
    -o cache/step3