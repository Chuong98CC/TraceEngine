# RGB-D Visualization (`tools/general_test/visualize_rgbd.py`)

Visualizes RGB-D data from **one of two mutually exclusive sources**:

1. **Camera mode** (`--camera_name`) — an Astribot camera frame loaded via
   `utils.astribot_dataloader.load_rgbd` (e.g. `head_rgbd`); metric depth is
   recovered from the greyscale depth image (clip at `--max_depth_m`).
   Sample data are in `assets/astribot_test_imgs`.
2. **Folder mode** (`--rgb_dir` + `--depth_npz_dir`) — pairs RGB images
   (`<stem>.jpg/.jpeg/.png`) by stem with a depth `<stem>.lz4` (raw uint16 mm,
   see `utils.astribot_dataloader.load_depth_lz4`, reshaped to the RGB frame
   size) and a pose `<stem>.npz` (`extrinsics` 3×4/4×4, `intrinsics` 3×3) —
   the format written by `run_stream.py` ([`streaming.md`](streaming.md)).
   `--frame_index` selects the pair at that position in the sorted matching
   stems.

```bash
# Astribot camera frame
python tools/general_test/visualize_rgbd.py --camera_name head_rgbd \
    --frame_index 0 --save_viz --save_glb

# RGB + depth-lz4 folder pair (e.g. a run_stream.py output)
python tools/general_test/visualize_rgbd.py \
    --rgb_dir path/to/rgb --depth_npz_dir path/to/depth \
    --save_viz --save_glb
```

| Argument | Default | Description |
|---|---|---|
| `--camera_name` | — | Astribot camera (mutually exclusive with the folder pair) |
| `--rgb_dir` / `--depth_npz_dir` | — | Folder pair: RGB images + depth `.lz4` / pose `.npz` (mutually exclusive with `--camera_name`) |
| `--frame_index` | `0` | Frame to process (camera mode: index; folder mode: position in sorted stems) |
| `--max_depth_m` | `5.0` | Far-plane clip for recovering metric depth from the greyscale image |
| `--output`, `-o` | `output/rgbd` | Output directory |
| `--save_viz` / `--save_glb` | off | Save the heatmap depth image or save `.glb` point cloud |
