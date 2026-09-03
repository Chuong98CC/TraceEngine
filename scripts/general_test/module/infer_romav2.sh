IMG1=assets/matching_points/coffee1.png
IMG2=assets/matching_points/coffee2.png
IMG3=assets/matching_points/coffee3.png
IMG4=assets/matching_points/coffee4.png
python tools/general_test/module/infer_romav2.py \
    $IMG1 $IMG2 $IMG3 $IMG4 \
    --strategy reference \
    --num-corresp 2000 --top-k 128 \
    --model weights/romav2/romav2.pt2 \
    --out cache/multi_match_matches.npz \
    --viz cache/multi_match_vis.jpg