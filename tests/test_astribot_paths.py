"""Layout invariants of the episodes/ tree."""
from pathlib import Path

from utils.astribot_paths import (
    DEPTH_POSE,
    episode_dir,
    episode_name,
    episodes_root,
    frame_stem,
    parse_episode,
    parse_subtask,
    subtask_dir,
    subtask_json,
    subtask_name,
    task_dir,
    depth_pose_dir,
    discover_cameras,
    discover_episodes,
    discover_subtasks,
)


def test_names():
    assert episode_name(0) == "ep000"
    assert episode_name(12) == "ep012"
    assert subtask_name(3) == "subtask_03"
    assert frame_stem(210) == "frame_000210"
    assert parse_episode("ep012") == 12
    assert parse_episode("ep0123") is None
    assert parse_episode("ep000000") is None
    assert parse_subtask("subtask_04") == 4
    assert parse_subtask("subtask_frames") is None


def test_tree_shape(tmp_path):
    root = episodes_root("/data/ds", out_dir=None)
    assert root == Path("/data/ds/episodes")
    assert episodes_root("/data/ds", out_dir="/tmp/out") == Path("/tmp/out")

    ep = episode_dir(root, 3)
    assert ep == Path("/data/ds/episodes/ep003")
    assert subtask_json(root, 3) == ep / "subtask.json"
    assert subtask_dir(root, 3, 0) == ep / "subtask_00"
    assert task_dir(root, 3, 0, DEPTH_POSE, "cam_head") == \
        ep / "subtask_00" / "depth_pose" / "cam_head"
    assert task_dir(root, 3, 0, DEPTH_POSE) == ep / "subtask_00" / "depth_pose"
    assert depth_pose_dir(root, 3, 0, "cam_head") == \
        ep / "subtask_00" / "depth_pose" / "cam_head"


def test_discovery(tmp_path):
    root = tmp_path
    (root / "ep000" / "subtask_00" / "depth_pose" / "cam_head").mkdir(parents=True)
    (root / "ep000" / "subtask_01" / "depth_pose" / "cam_head").mkdir(parents=True)
    (root / "ep000" / "subtask_01" / "depth_pose" / "cam_torso").mkdir(parents=True)
    (root / "ep002" / "subtask_00").mkdir(parents=True)
    (root / "not_an_episode").mkdir()

    assert discover_episodes(root) == [0, 2]
    assert discover_subtasks(root, 0) == [0, 1]
    assert discover_cameras(root, 0, 1, DEPTH_POSE) == ["cam_head", "cam_torso"]
    assert discover_cameras(root, 0, 0, "traces") == []
