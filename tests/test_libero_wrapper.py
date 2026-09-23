"""CPU-only tests for the LIBERO LeRobot adapter.

Only the last test reads a dataset.  Everything else drives the wrapper with a
hand-built episode table and a stand-in dataset, because the wrappers are
metadata plumbing whose failure mode is a wrong column mapping — a LeRobot
version bump, not a decoding bug — and the frame path's contract (absolute
indices in, ``(H, W, 3)`` uint8 out, one camera key read) is checkable against
synthetic frames without pulling in torchcodec or the videos.

The real copies carry their episode table as a HuggingFace ``datasets`` table,
which indexes exactly like a dict of lists (``table[col][idx]``), so the dict
layout below stands in for it; pandas is the layout that takes a different
path through the adapter and is covered on its own.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from utils import libero_wrapper
from utils.libero_wrapper import LiberoWrapper

REPO_ID = "local/libero_goal"
CAMERA = "observation.images.image"
OTHER_CAMERA = "observation.images.wrist_image"
FPS = 20.0

# (episode_index, from_index, to_index, length, from_timestamp, to_timestamp,
# task) of three contiguous episodes whose timestamps match their frame counts
ROWS = (
    (0, 0, 138, 138, 0.0, 6.9, "open the middle drawer"),
    (1, 138, 268, 130, 6.9, 13.4, "close the drawer"),
    (2, 268, 400, 132, 13.4, 20.0, "pick up the box"),
)

# Frames stay tiny: these tests exercise the conversion, not the renders.  The
# tensor holds k / 255, which is what the dataset serves, so an exact
# round-trip is k.
H, W = 2, 4
FRAME_TENSOR = torch.arange(3 * H * W, dtype=torch.float32).reshape(3, H, W) / 255.0
FRAME_UINT8 = np.arange(3 * H * W, dtype=np.uint8).reshape(3, H, W).transpose(1, 2, 0)

DATA_ROOT = (Path(__file__).resolve().parents[1] / "data" / "libero_mujoco3.3.2"
             / "lerobot_v30" / "libero_goal_no_noops_lerobot")


def table(layout="dict"):
    """The episode table in the requested layout."""
    cols = {
        "episode_index": [row[0] for row in ROWS],
        "dataset_from_index": [row[1] for row in ROWS],
        "dataset_to_index": [row[2] for row in ROWS],
        "length": [row[3] for row in ROWS],
        "tasks": [[row[6]] for row in ROWS],  # v3.0 stores a list of them
        f"videos/{CAMERA}/from_timestamp": [row[4] for row in ROWS],
        f"videos/{CAMERA}/to_timestamp": [row[5] for row in ROWS],
    }
    if layout == "dict":
        return cols
    if layout == "pandas":
        pd = pytest.importorskip("pandas", reason="the pandas layout needs pandas")
        return pd.DataFrame(cols)
    raise AssertionError(f"unknown layout {layout!r}")


class FakeMetadata:
    """Stand-in for ``LeRobotDatasetMetadata``: the episode table plus the
    attributes the wrapper reads off the metadata."""

    def __init__(self, episodes, camera_keys=(CAMERA, OTHER_CAMERA)):
        self.episodes = episodes
        self.camera_keys = list(camera_keys)
        self.fps = FPS
        # the row count comes from a column: a dict of lists is as long as its
        # column count, not as long as its table
        self.total_episodes = len(episodes["episode_index"])


class RecordingRow(dict):
    """One dataset row that notes which of its keys are read."""

    def __init__(self, keys_read):
        super().__init__({CAMERA: FRAME_TENSOR.clone(),
                          OTHER_CAMERA: FRAME_TENSOR.clone()})
        self._keys_read = keys_read

    def __getitem__(self, key):
        self._keys_read.add(key)
        return super().__getitem__(key)


class FakeDataset:
    """Stand-in for ``LeRobotDataset``: serves one synthetic frame per dataset
    index, recording every index asked for so a test can check the wrapper
    decoded what it promised."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.requested = []
        self.keys_read = set()

    def __getitem__(self, index):
        self.requested.append(index)
        return RecordingRow(self.keys_read)


def patch_lerobot(monkeypatch, episodes):
    """Swap both LeRobot entry points for the fake episode table.  The dataset
    becomes a landmine, so a test that reaches it before replacing it fails
    loudly instead of touching the disk."""
    monkeypatch.setattr(libero_wrapper, "LeRobotDatasetMetadata",
                        lambda **kwargs: FakeMetadata(episodes))
    monkeypatch.setattr(libero_wrapper, "LeRobotDataset", _no_dataset)


def _no_dataset(*args, **kwargs):
    raise AssertionError("LeRobotDataset must not be constructed")


def make(monkeypatch, episodes=None, camera=CAMERA, metadata_only=False):
    """A wrapper over the fake table."""
    patch_lerobot(monkeypatch, table() if episodes is None else episodes)
    return LiberoWrapper(REPO_ID, "/nonexistent/libero_goal", camera,
                         metadata_only=metadata_only)


def make_decoding(monkeypatch, episodes=None):
    """``(wrapper, built)`` where ``built`` collects the fake datasets the
    wrapper constructs — at most one, on the first ``frames()`` call."""
    built = []

    def _factory(**kwargs):
        dataset = FakeDataset(**kwargs)
        built.append(dataset)
        return dataset

    wrapper = make(monkeypatch, episodes)
    monkeypatch.setattr(libero_wrapper, "LeRobotDataset", _factory)
    assert built == []  # constructing a wrapper must not open the dataset
    return wrapper, built


@pytest.mark.parametrize("layout", ["dict", "pandas"])
def test_episodes_map_table_columns(monkeypatch, layout):
    """Every field of an Episode comes from its own table column, and the
    half-open frame range, the length and the timestamps agree with each
    other."""
    wrapper = make(monkeypatch, table(layout))
    assert wrapper.name == "libero_goal"
    assert wrapper.fps == FPS
    assert wrapper.total_episodes == 3
    assert wrapper.camera_keys == [CAMERA, OTHER_CAMERA]

    episodes = wrapper.episodes()
    assert [e.index for e in episodes] == [0, 1, 2]
    assert [(e.from_index, e.to_index) for e in episodes] == [
        (0, 138), (138, 268), (268, 400)]
    assert [e.length for e in episodes] == [138, 130, 132]
    assert [e.task for e in episodes] == [row[6] for row in ROWS]
    assert [(e.from_timestamp, e.to_timestamp) for e in episodes] == [
        (0.0, 6.9), (6.9, 13.4), (13.4, 20.0)]
    for episode in episodes:
        assert episode.to_index - episode.from_index == episode.length
        assert wrapper.frame_count(episode) == episode.length
        assert episode.to_timestamp - episode.from_timestamp == pytest.approx(
            episode.length / FPS)

    with pytest.raises(dataclasses.FrozenInstanceError):
        episodes[0].length = 1


def test_episode_selection_keeps_the_requested_order(monkeypatch):
    """``idxes`` name episodes, and the result follows the caller's order —
    a reordered job list must not come back silently sorted."""
    wrapper = make(monkeypatch)
    assert [e.index for e in wrapper.episodes([2, 0])] == [2, 0]
    assert [e.index for e in wrapper.episodes([1])] == [1]
    assert [e.task for e in wrapper.episodes([2])] == [ROWS[2][6]]
    assert wrapper.episodes([]) == []


def test_unknown_episode_raises(monkeypatch):
    """An index that names no episode is an error, not a shorter list."""
    wrapper = make(monkeypatch)
    with pytest.raises(IndexError) as err:
        wrapper.episodes([0, 7])
    assert "episode 7" in str(err.value)
    assert "3 episodes" in str(err.value)


def test_unknown_camera_names_the_available_ones(monkeypatch):
    """A bad camera key fails on construction, with the keys to choose from."""
    with pytest.raises(ValueError) as err:
        make(monkeypatch, camera="observation.images.head")
    message = str(err.value)
    assert "observation.images.head" in message
    assert CAMERA in message
    assert OTHER_CAMERA in message


def test_metadata_only_never_builds_the_dataset(monkeypatch):
    """Enumeration works with the dataset entry point left as a landmine."""
    wrapper = make(monkeypatch, metadata_only=True)
    assert [e.index for e in wrapper.episodes()] == [0, 1, 2]
    with pytest.raises(RuntimeError) as err:
        wrapper.frames(wrapper.episodes()[0], [0])
    assert "metadata_only" in str(err.value)
    assert wrapper._dataset is None


def test_frames_decode_to_uint8_hwc(monkeypatch):
    """Frames come back ``(H, W, 3)`` uint8 RGB, exactly ``round(x * 255)`` of
    the dataset's float tensor, paired with the absolute dataset index."""
    wrapper, built = make_decoding(monkeypatch)
    got = list(wrapper.frames(wrapper.episodes()[0], [0, 2]))

    assert [index for index, _ in got] == [0, 2]
    for _, frame in got:
        assert frame.shape == (H, W, 3)
        assert frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
        assert np.array_equal(frame, FRAME_UINT8)

    assert len(built) == 1  # opened once, on the first frames() call
    assert built[0].requested == [0, 2]  # decoded what was asked, in order
    assert built[0].keys_read == {CAMERA}  # and only the one camera key
    assert built[0].init_kwargs["download_videos"] is False


def test_frames_default_to_the_whole_episode(monkeypatch):
    """Without ``indices`` every frame of the episode is decoded, in order,
    under its absolute dataset index."""
    wrapper, built = make_decoding(monkeypatch)
    episode = wrapper.episodes()[1]  # [138, 268)
    got = list(wrapper.frames(episode))

    assert [index for index, _ in got] == list(range(138, 268))
    assert len(got) == wrapper.frame_count(episode) == 130
    assert built[0].requested == list(range(138, 268))


def test_frames_reject_indices_outside_the_episode(monkeypatch):
    """Episode-relative indices are the mistake the signature invites, so they
    are rejected up front — before the dataset is opened."""
    wrapper, built = make_decoding(monkeypatch)
    episode = wrapper.episodes()[1]  # [138, 268)
    for bad in ([0], [137], [268], [137, 138]):
        with pytest.raises(IndexError) as err:
            wrapper.frames(episode, bad)
        assert "absolute" in str(err.value)
    assert built == []


@pytest.mark.skipif(not DATA_ROOT.is_dir(),
                    reason=f"no LIBERO v3.0 copy at {DATA_ROOT}")
def test_real_dataset_decodes_frames():
    """The one test that reads real bytes: a couple of frames of the first and
    last episodes decode to the dataset's ``(H, W, 3)`` uint8, and the camera
    timestamps describe the same span as the frame count (within a frame, the
    resolution the table stores them at)."""
    wrapper = LiberoWrapper(REPO_ID, DATA_ROOT)
    first = wrapper.episodes()[0]
    assert first.task
    assert wrapper.frame_count(first) == first.length
    assert first.to_timestamp - first.from_timestamp == pytest.approx(
        first.length / wrapper.fps, abs=1.0 / wrapper.fps)

    shape = tuple(wrapper.metadata.features[CAMERA]["shape"][:2])  # (H, W)
    got = list(wrapper.frames(first, [first.from_index, first.from_index + 1]))
    assert [index for index, _ in got] == [first.from_index, first.from_index + 1]
    for _, frame in got:
        assert frame.shape == (*shape, 3)
        assert frame.dtype == np.uint8
        assert frame.max() > 0  # decoded pixels, not an empty buffer

    # a frame from another episode: the index space spans the whole dataset,
    # so seeking past the first video file has to work
    last = wrapper.episodes()[-1]
    (index, frame), = wrapper.frames(last, [last.from_index])
    assert index == last.from_index
    assert frame.dtype == np.uint8
