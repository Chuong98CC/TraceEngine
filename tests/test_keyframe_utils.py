"""Step-1 metadata access over the merged subtask.json."""
import json

import pytest

from utils.keyframe_utils import load_subtask, load_subtask_labels


def _write(root, ep_idx, payload):
    from utils.astribot_paths import episode_dir
    ep = episode_dir(root, ep_idx)
    ep.mkdir(parents=True, exist_ok=True)
    (ep / "subtask.json").write_text(json.dumps(payload))


def test_load_subtask_roundtrip(tmp_path):
    _write(tmp_path, 3, {
        "episode": 3, "task_id": 0, "task": "Make coffee",
        "from_idx": 0, "to_idx": 100,
        "key_frames": [0, 50, 99], "split_frames": [50],
        "labels": [0, 2, None],
    })
    data = load_subtask(tmp_path, 3)
    assert data["split_frames"] == [50]
    assert load_subtask_labels(tmp_path, 3) == [0, 2, None]


def test_labels_accessor_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subtask_labels(tmp_path, 9)
