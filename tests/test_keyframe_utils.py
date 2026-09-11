"""Step-1 metadata access over the merged subtask.json — the readers Step 3
uses (utils.keyframe_utils), plus the pairing of that file's
``subtask_labels`` with
the segmentation the reading modes lay the ``subtask_XX`` dirs out with
(tools/astribot/extract_frames.py)."""
import argparse
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
        "subtask_labels": [0, 2, None],
    })
    data = load_subtask(tmp_path, 3)
    assert data["split_frames"] == [50]
    assert load_subtask_labels(tmp_path, 3) == [0, 2, None]


def test_labels_accessor_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_subtask_labels(tmp_path, 9)


# --- the labels must describe the reading modes' segmentation ----------------

#: one sub-task per 20 frames, executed in the order 0, 2, 1, 3, 5 — the
#: ground truth of the frame table's subtask_index column, which the reading
#: modes segment by (see _load_splits).
N_FRAMES = 100
GROUND_TRUTH_RUNS = [0] * 20 + [2] * 20 + [1] * 20 + [3] * 20 + [5] * 20


def _detect(tmp_path, subtask_index, inferred_splits, use_inferred_splits):
    """Run one detect_subtask episode and return the subtask.json it wrote.

    Only the dataset I/O is stubbed: the episode frame table and the
    gripper-inferred split frames (a separate analysis with its own coverage)
    are handed in, while _process_episode, _resolve_segment_labels,
    _split_frames_from_ground_truth and _save_subtask_json run for real.
    """
    import pandas as pd

    from tools.astribot.extract_frames import DataExtract

    columns = {"index": list(range(N_FRAMES))}
    if subtask_index is not None:
        columns["subtask_index"] = subtask_index
    df = pd.DataFrame(columns)
    ex = object.__new__(DataExtract)  # __init__ would open the dataset
    ex.args = argparse.Namespace(
        mode="detect_subtask", data_root=str(tmp_path), out_dir=None,
        use_inferred_splits=use_inferred_splits, dedup_tasks=False,
        min_close_seconds=1.5, interval=4)
    ex._root = ex.out_dir = tmp_path
    ex.tasks = None
    ex.gripper_idxes = []  # skips the gripper plot
    ex.done_tasks = set()
    ex.fps = 30

    def _begin(ep_idx):
        ex.ep_idx, ex.from_idx = ep_idx, 0
        ex.to_idx, ex.task_id = len(df), 0
        return df

    ex._begin_episode = _begin
    ex._load_episode_table = lambda: df
    ex._subtask_split_idxes = lambda *args, **kwargs: list(inferred_splits)
    ex._process_episode(0)
    return json.loads((tmp_path / "ep000" / "subtask.json").read_text())


def test_labels_follow_the_ground_truth_segmentation(tmp_path):
    # 5 ground-truth segments against 2 inferred ones: the reading modes lay
    # the dirs out by ground truth, so a 2-label list would leave the last
    # three segments unlabelled and prompt the first two from the wrong
    # segmentation
    data = _detect(tmp_path, GROUND_TRUTH_RUNS, inferred_splits=[50],
                   use_inferred_splits=False)
    assert data["split_frames"] == [50]  # the inferred splits still land in the file
    assert data["subtask_labels"] == [0, 2, 1, 3, 5]


def test_labels_follow_the_inferred_segmentation_when_preferred(tmp_path):
    # --use-inferred-splits makes the reading modes consume the inferred
    # splits, so those are the bounds the labels must be paired with
    data = _detect(tmp_path, GROUND_TRUTH_RUNS, inferred_splits=[50],
                   use_inferred_splits=True)
    assert data["split_frames"] == [50]
    assert data["subtask_labels"] == [0, 2]


def test_labels_fall_back_to_the_inferred_splits_without_annotations(tmp_path):
    # no subtask_index column: _load_splits falls through to the inferred
    # splits, and with no annotation order either the ordinals are assumed
    data = _detect(tmp_path, None, inferred_splits=[50],
                   use_inferred_splits=False)
    assert data["split_frames"] == [50]
    assert data["subtask_labels"] == [0, 1]
