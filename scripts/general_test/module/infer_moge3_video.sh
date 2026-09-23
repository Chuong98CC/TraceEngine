# MoGe v3 over the episodes of a local LeRobot dataset -> one octahedral-packed
# lossless FFV1 video per episode-camera, plus a .npz sidecar beside it (the
# packed frames cannot be decoded without it). Each episode lands at
# OUT_DIR/<dataset>/<camera>/episode_%06d.mkv, which is what keeps same-named
# episodes of different datasets and cameras apart.
#
# The dataset runs at ~67 ms/frame (512x512 export) over 433 episodes, so
# budget hours rather than minutes. Add --limit to smoke-test, --skip-existing
# to resume, and --dry-run to size the job without loading the model.
REPO_ID='local/libero_goal_no_noops_lerobot'   # never fetched: any stable id
DATA_ROOT='data/libero_mujoco3.3.2/lerobot_v30/libero_goal_no_noops_lerobot'
CAMERA='observation.images.image'
OUT_DIR='data/libero_moge3_octa'

python tools/general_test/module/infer_moge3_video.py \
    --repo-id $REPO_ID \
    --data-root $DATA_ROOT \
    --camera $CAMERA \
    --out-dir $OUT_DIR \
    --pt2 weights/moge3/moge3_l_512x512.pt2 \
    --input-size 512 512 --refine_steps 2 \
    --range-sample-frames 8 \
    --skip-existing
    # --save-viz --limit 3 --max-frames 150         # small first run
    # --save-viz \                      # a small side-by-side mp4 per episode
    #                                   # (input | packed output) to look at;
    #                                   # ~40x smaller than the FFV1. --viz-crf
    #                                   # tunes the size
    # --align \                         # off by default, and worth leaving
    #                                   # off: it measures better on the
    #                                   # background but reads as shakier.
    #                                   # Needs a fixed camera (refused on
    #                                   # 'wrist' by name)
    # --depth-max 4.0 \                 # scenes past 3 m are clipped; raising
    #                                   # this is free for nearer ones, since
    #                                   # the range still tightens to the data
    # --stride 2 \                      # every 2nd frame, written at fps/2
