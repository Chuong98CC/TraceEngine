"""Read-only adapter over a local LeRobot dataset copy for the LIBERO tools.

Two facts about the LIBERO copies under ``data/libero_mujoco3.3.2`` are worth
not rediscovering:

* **Only the LeRobot v3.0 copies are readable.**  The tree holds a v2.1 copy of
  every dataset next to ``lerobot_v30/``; lerobot 0.6.2 refuses to open those
  (``BackwardCompatibilityError``), so ``data_root`` must point at the
  directory *under* ``lerobot_v30/``.
* **OpenCV cannot decode these videos.**  They are AV1, which the local cv2
  build opens but never gets a frame out of (``read()`` returns False), so
  ``LeRobotDataset.__getitem__`` — the LeRobot/torchcodec path taken here — is
  the only way to get pixels out of them.

:class:`LiberoWrapper` puts the two LeRobot entry points a tool needs behind
one object: ``LeRobotDatasetMetadata`` for episode enumeration and
``LeRobotDataset`` for frame decode.  The dataset handle is built on the first
:meth:`LiberoWrapper.frames` call and never before, so enumerating episodes
costs no video machinery and works on a machine that holds only the ``meta/``
tree.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata


@dataclass(frozen=True)
class Episode:
    """One episode of a dataset, in the terms a video tool needs it.

    ``from_index``/``to_index`` are absolute dataset indices and the range is
    half-open — the episode owns exactly ``[from_index, to_index)``.
    ``from_timestamp``/``to_timestamp`` are the chosen camera's own timestamps
    from the episode table, in seconds; they describe the same span as the
    frame range (``to_timestamp - from_timestamp == length / fps`` on the
    shipped copies), but only the timestamps are what the video files are
    seeked with, so both are carried rather than one derived from the other.

    Frozen because the tools hand episodes around between stages (a job list,
    a worker) and a mutated episode would silently address different frames in
    each.
    """

    index: int
    from_index: int
    to_index: int
    length: int
    from_timestamp: float
    to_timestamp: float
    task: str


class LiberoWrapper:
    """Episode metadata and frame decode for one local LeRobot v3.0 dataset.

    ``data_root`` is the dataset directory itself (the one holding ``meta/``,
    ``data/`` and ``videos/`` — e.g. ``.../lerobot_v30/libero_goal_no_noops_lerobot``),
    not the tree of copies.  ``repo_id`` is only used to satisfy the LeRobot
    constructors; nothing is fetched from the Hub, so any stable id works as
    long as it is the same one for the metadata and the dataset.

    ``camera`` is the feature key to decode (``observation.images.image`` or
    ``observation.images.wrist_image`` on these datasets).  It is validated
    against the dataset's camera keys on construction, because an unknown key
    would otherwise only surface much later, as a ``KeyError`` from deep
    inside the first decoded frame.

    ``metadata_only=True`` builds a wrapper that refuses to decode
    (:meth:`frames` raises) — the mode for enumerating episodes on a machine
    that has the metadata but not the videos.
    """

    def __init__(
        self,
        repo_id: str,
        data_root: str | Path,
        camera: str = "observation.images.image",
        *,
        metadata_only: bool = False,
    ) -> None:
        self.repo_id = repo_id
        self.data_root = Path(data_root)
        self.camera = camera
        self.metadata_only = metadata_only
        self.metadata = LeRobotDatasetMetadata(repo_id=repo_id,
                                               root=str(self.data_root))
        self.total_episodes = int(self.metadata.total_episodes)
        self.fps = float(self.metadata.fps)
        keys = self.camera_keys
        if camera not in keys:
            raise ValueError(
                f"camera {camera!r} is not in {self.name!r}; available: "
                + ", ".join(repr(key) for key in keys)
            )
        self._dataset = None  # LeRobotDataset handle, built by _ensure_dataset
        self._episodes = None  # cached episode list, built by _build_episodes

    @property
    def name(self) -> str:
        """Directory name of the dataset, which the tools use to name
        outputs."""
        return self.data_root.name

    @property
    def camera_keys(self) -> list[str]:
        """The dataset's camera keys, in metadata order."""
        return list(self.metadata.camera_keys)

    # --- episode table --------------------------------------------------------

    @staticmethod
    def _col_names(episodes) -> list[str]:
        """Column names of an episode table: ``.columns`` for a pandas
        DataFrame, ``.column_names`` for the HuggingFace ``datasets`` table
        LeRobot actually uses, else the keys of a dict of lists."""
        for attr in ("columns", "column_names"):
            names = getattr(episodes, attr, None)
            if names is not None:
                return list(names)
        return list(episodes.keys())

    @staticmethod
    def _cell(episodes, idx: int, col: str):
        """One cell of an episode table.

        The table layout depends on what wrote the copy, and all three layouts
        in the wild index a column the same way except pandas: a HuggingFace
        ``datasets`` table hands back a column object that indexes by position
        (``table[col][idx]``), a dict of lists is the same expression, and only
        a pandas DataFrame needs the ``.iloc`` hop.
        """
        if hasattr(episodes, "columns"):  # pandas DataFrame
            return episodes[col].iloc[idx]
        return episodes[col][idx]

    @staticmethod
    def _task(episodes, idx: int) -> str:
        """Task description of an episode.

        LeRobot v3.0 keeps it in the ``tasks`` column as a list of
        instructions, usually of length one; a task that spans several
        instructions is joined rather than truncated, so nothing is dropped.
        """
        val = LiberoWrapper._cell(episodes, idx, "tasks")
        if isinstance(val, (list, tuple, np.ndarray)):
            return "; ".join(str(part) for part in val)
        return "" if val is None else str(val)

    def _build_episodes(self) -> list[Episode]:
        """Every episode of the dataset, in table order.

        The row count comes from a column rather than ``len(table)``: a
        dict-of-lists table's length is its column count, not its row count.
        """
        episodes = self.metadata.episodes
        cols = self._col_names(episodes)
        ts_cols = (f"videos/{self.camera}/from_timestamp",
                   f"videos/{self.camera}/to_timestamp")
        missing = [col for col in ts_cols if col not in cols]
        if missing:
            raise ValueError(
                f"camera {self.camera!r} has no per-episode timestamps in "
                f"{self.name!r} (missing {', '.join(missing)}); the episode "
                f"table has: {', '.join(cols)}"
            )
        n_rows = len(episodes["dataset_from_index"])
        has = {col: col in cols for col in ("episode_index", "length", "tasks")}
        out = []
        for i in range(n_rows):
            from_index = int(self._cell(episodes, i, "dataset_from_index"))
            to_index = int(self._cell(episodes, i, "dataset_to_index"))
            out.append(Episode(
                index=(int(self._cell(episodes, i, "episode_index"))
                       if has["episode_index"] else i),
                from_index=from_index,
                to_index=to_index,
                # the length column is authoritative, but a table without one
                # describes the same span through the bounds
                length=(int(self._cell(episodes, i, "length"))
                        if has["length"] else to_index - from_index),
                from_timestamp=float(self._cell(episodes, i, ts_cols[0])),
                to_timestamp=float(self._cell(episodes, i, ts_cols[1])),
                task=self._task(episodes, i) if has["tasks"] else "",
            ))
        return out

    def episodes(self, idxes: Sequence[int] | None = None) -> list[Episode]:
        """The dataset's episodes, all of them or the requested ones.

        ``idxes`` are episode indices (the ``episode_index`` column), not row
        positions, and the result keeps the order they were asked in, so a
        caller that reorders or dedups a job list gets its own order back.
        An index that names no episode raises ``IndexError`` rather than
        yielding a shorter list: a silently dropped episode looks to a caller
        like a dataset that ended early.
        """
        if self._episodes is None:
            self._episodes = self._build_episodes()
        if idxes is None:
            return list(self._episodes)
        by_index = {episode.index: episode for episode in self._episodes}
        out = []
        for i in idxes:
            episode = by_index.get(int(i))
            if episode is None:
                raise IndexError(
                    f"episode {i} is not in {self.name!r}, which has "
                    f"{self.total_episodes} episodes"
                )
            out.append(episode)
        return out

    def frame_count(self, episode: Episode) -> int:
        """Number of frames the episode holds (its ``length`` column)."""
        return episode.length

    # --- frame decode ---------------------------------------------------------

    def frames(
        self,
        episode: Episode,
        indices: Sequence[int] | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Decode frames of an episode, as ``(dataset_index, frame)`` pairs.

        Every yielded frame is ``(H, W, 3)`` uint8 RGB numpy.  The dataset
        serves it as a ``(3, H, W)`` float tensor in [0, 1]; the conversion is
        a plain ``round(x * 255)``, the exact inverse of the ``x / 255`` the
        dataset applied, so the bytes are what the video holds.

        ``indices`` are absolute dataset indices — the same space as
        ``Episode.from_index``/``to_index``, not episode-relative ones — and
        default to every frame of the episode in order.  Indices outside the
        episode raise ``IndexError`` before anything is decoded, since a
        wrong-space index (an episode-relative one) is the mistake this
        signature invites.
        """
        if self.metadata_only:
            raise RuntimeError(
                f"{self.name!r} was opened with metadata_only=True, so it has "
                "no video access; build the wrapper without that flag to "
                "decode frames"
            )
        if indices is None:
            idxes = list(range(episode.from_index, episode.to_index))
        else:
            idxes = [int(i) for i in indices]
        outside = [i for i in idxes
                   if not episode.from_index <= i < episode.to_index]
        if outside:
            raise IndexError(
                f"{len(outside)} index(es) fall outside episode "
                f"{episode.index} of {self.name!r}, whose frames are "
                f"[{episode.from_index}, {episode.to_index}): {outside[0]} "
                "and others — indices are absolute dataset indices, not "
                "episode-relative"
            )
        return self._iter_frames(self._ensure_dataset(), idxes)

    def _iter_frames(self, dataset, idxes: Sequence[int]):
        """Yield ``(dataset_index, frame)`` for already-validated indices."""
        for i in idxes:
            yield i, self._to_hwc_uint8(dataset[i][self.camera])

    @staticmethod
    def _to_hwc_uint8(frame: torch.Tensor) -> np.ndarray:
        """``(H, W, 3)`` uint8 RGB numpy of a ``(3, H, W)`` float frame.

        The result is made contiguous: the channel permute alone would hand
        back a strided view, and the tools downstream feed these arrays to
        writers that expect a plain buffer.
        """
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"expected a (3, H, W) RGB frame, got shape "
                f"{tuple(frame.shape)}"
            )
        return (frame.detach().to(torch.float32).mul(255).round()
                .clamp(0, 255).to(torch.uint8)
                .permute(1, 2, 0).contiguous().numpy())

    def _ensure_dataset(self) -> LeRobotDataset:
        """The ``LeRobotDataset`` handle, built on first use — constructing it
        sets up the video decode machinery, which only this path needs, and
        which needs the videos to be on disk."""
        if self._dataset is None:
            self._dataset = LeRobotDataset(repo_id=self.repo_id,
                                           root=str(self.data_root),
                                           download_videos=False)
        return self._dataset
