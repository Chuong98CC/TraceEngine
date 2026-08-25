
import os
from tqdm import tqdm
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from utils.visualize_mask import to_pil
import random
repo_id = "simple-world-lab/HiFi-UMI-2K"
chunk_idx="0000"
data_root=f"/data/HiFi-UMI-2K/chunk-{chunk_idx}/part-0000"
cam_idxes = [0,1]
# save_dir = f"{data_root}/key_frames"
save_dir = "cache/hifi-umi"
os.makedirs(save_dir, exist_ok=True)
ds_meta = LeRobotDatasetMetadata(repo_id=repo_id, root=data_root)
dataset = LeRobotDataset(repo_id=repo_id, root=data_root, download_videos=False)
total_eps = ds_meta.total_episodes
num_eps= 10
num_frames = ds_meta.total_frames
eps = ds_meta.episodes
head_left_cam= ds_meta.camera_keys[0]
head_right_cam= ds_meta.camera_keys[1]
ranom_eps = random.sample(range(total_eps), num_eps)
# for ep_idx in tqdm(ranom_eps):
task_done = set()
for ep_idx in tqdm(range(total_eps)):
    from_idx = eps["dataset_from_index"][ep_idx]
    frame = dataset[from_idx]
    task_id = frame["task_index"]
    if task_id.item() in task_done:
        continue
    task_done.add(task_id.item())
    frame_idx = from_idx
    cam_key = ds_meta.camera_keys[cam_idxes[0]]
    cam_img = frame[cam_key]
    to_pil(cam_img).save(f"{save_dir}/task{task_id.item()}_ep{ep_idx}_frame_{frame_idx:06d}.jpg")
print(task_done)
