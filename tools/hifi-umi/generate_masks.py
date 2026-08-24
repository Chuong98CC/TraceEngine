#%%
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
import json
import matplotlib.pyplot as plt
import torch
import numpy as np
# turn on tfloat32 for Ampere GPUs (also done by sam3_runtime on import)
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# use bfloat16 for the entire script (the model class also wraps its runs and
# post-processing in the trace-matching autocast)
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

from det_seg_models.sam3 import (
    Sam3Image,
    plot_results
)

#%%
# Load models
repo_id = "simple-world-lab/HiFi-UMI-2K"
chunk_idx="0000"
data_root=f"/data/HiFi-UMI-2K/chunk-{chunk_idx}/part-0000"
folder_root="/home/chuong/workspace/depth_models/DepthModels"
task_parse_file = f"{folder_root}/data/hifi-umi/task_parsed/chunk-{chunk_idx}.json"
sam3_ckpt=f"{folder_root}/weights/sam3/sam3_image_exported_bf16.pt2"
task_parsed = json.load(open(task_parse_file, "r"))
ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=data_root)
dataset = LeRobotDataset(repo_id=repo_id, root=data_root, download_videos=False)
sam3 = Sam3Image(sam3_ckpt)
#%%
# num_eps = ds_meta.num_episodes
num_eps = 1
eps = ds_meta.episodes
for ep_idx in range(num_eps):
    from_idx = eps["dataset_from_index"][ep_idx]
    to_idx = eps["dataset_to_index"][ep_idx]

    # process mask for the from_idx and start_idx frames
    results = []
    for frame_idx in [from_idx, to_idx]:
        frame = dataset[frame_idx]
        # process the frame to generate mask
        left_head_img = frame["observation.images.head_main"]
        right_head_img = frame["observation.images.head_main_stereo_right"]
        # task id
        task_id = frame["task_index"]
        task_info = task_parsed[str(task_id.item())]
        object_prompt = task_info["object"]
        print(f"Processing: Episode{ep_idx}, frame {frame_idx}, task_id {task_id}, object_prompt {object_prompt}")
        for obj in object_prompt:
            #TODO: call SAM3 model with the text prompt `obj` to generate the mask for the object in the frame
            sam3_result = sam3.predict(left_head_img, text_prompt=obj)
            results.append({
                "frame_idx": frame_idx,
                "object": obj,
                "sam3_result": sam3_result
            })
#%%
# visualize results
from utils.visualize_mask import to_pil
# for result in results[:1]:
result = results[-1]
frame_idx = result["frame_idx"]
obj = result["object"]
sam3_result = result["sam3_result"]
frame = dataset[frame_idx]
left_head_img = frame["observation.images.head_main"]
img_pil = to_pil(left_head_img)
print(f"Visualizing: Frame {frame_idx} - Object: {obj}")
plot_results(img_pil, sam3_result)
# %%
sam3_result1 = sam3.predict(left_head_img, text_prompt="yellow object")
plot_results(img_pil, sam3_result1)

# %%
print(ds_meta.camera_keys)
# %%
frame_idx = 200
for cam_key in ds_meta.camera_keys:
    cam_img = dataset[frame_idx][cam_key]
    img_pil = to_pil(cam_img)
    print(f"Visualizing: Frame {frame_idx} - Camera: {cam_key}")
    plot_results(img_pil, sam3_result)
# %%
