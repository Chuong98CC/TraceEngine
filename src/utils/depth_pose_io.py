"""Indexed on-disk store for the Step-2 depth_pose outputs.

One (subtask, camera) folder holds two files:

    depth.lz4   appendable container: a 64-byte header, one length-prefixed
                lz4 record per frame (the same payload save_depth_lz4
                produces), then an offset index and a 16-byte trailer
                written at close().  Decoded frames are float32 metres, the
                codec range coming from the header.
    poses.npz   frame_indices / extrinsics / intrinsics / shape for every
                frame of the container, in container order.

The container is read three ways: sequentially front-to-back (one open, no
index needed), randomly by position via the offset index, and by scanning
the record framing of a file whose writer died before close() — the header
and records stay valid, only the index and poses.npz are missing.
"""

import os
import struct
from pathlib import Path
from typing import Optional

import lz4.frame
import numpy as np

from utils.depth_utils import MAX_DEPTH, MIN_DEPTH, LogDepthToUint8Transform

#: Container + pose-store file names inside a depth_pose camera folder.
DEPTH_FILE = "depth.lz4"
POSES_FILE = "poses.npz"

MAGIC = b"DPK1"
VERSION = 1
CODEC_LOG_UINT8 = 0  # LogDepthToUint8Transform over the header's [min, max]

HEADER_SIZE = 64
_HEADER = struct.Struct("<4sIIIIffIQQ16s")
_TRAILER = struct.Struct("<Q8s")
_LENGTH = struct.Struct("<Q")
TRAILER_MAGIC = b"DPK1IDX"
TRAILER_SIZE = _TRAILER.size


class DepthContainerWriter:
    """Append-only writer for one camera's depth container.

    Frames are appended in order as they are produced; close() writes the
    offset index and the trailer and patches the header, which is what makes
    the file randomly addressable.  A writer that never reaches close()
    leaves a sequentially readable file (see DepthContainerReader).
    """

    def __init__(self, path, height: int, width: int,
                 clip: tuple[float, float] = (MIN_DEPTH, MAX_DEPTH)):
        self.path = Path(path)
        self.height = int(height)
        self.width = int(width)
        self.clip = (float(clip[0]), float(clip[1]))
        self._codec = LogDepthToUint8Transform(min_depth_m=self.clip[0],
                                               max_depth_m=self.clip[1])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._offsets = [HEADER_SIZE]  # record starts; last entry = file end
        self._end = HEADER_SIZE
        self._closed = False
        self._write_header(frame_count=0, index_offset=0)

    def _write_header(self, frame_count: int, index_offset: int) -> None:
        self._fh.seek(0)
        self._fh.write(_HEADER.pack(
            MAGIC, VERSION, self.height, self.width, CODEC_LOG_UINT8,
            self.clip[0], self.clip[1], 0, frame_count, index_offset,
            b"\x00" * 16,
        ))

    def append(self, depth_m: np.ndarray) -> None:
        """Encode + compress + append one (H, W) depth map in metres."""
        encoded = self._codec.encode(np.asarray(depth_m))
        if encoded.shape != (self.height, self.width):
            raise ValueError(
                f"depth shape {encoded.shape} does not match the container "
                f"({self.height}, {self.width})"
            )
        payload = lz4.frame.compress(encoded.tobytes())
        self._fh.seek(self._end)
        self._fh.write(_LENGTH.pack(len(payload)))
        self._fh.write(payload)
        self._end += _LENGTH.size + len(payload)
        self._offsets.append(self._end)

    @property
    def frame_count(self) -> int:
        return len(self._offsets) - 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def close(self) -> None:
        if self._closed:
            return
        index_offset = self._end
        self._fh.seek(index_offset)
        self._fh.write(np.asarray(self._offsets, dtype="<u8").tobytes())
        self._fh.write(_TRAILER.pack(index_offset, TRAILER_MAGIC))
        self._write_header(frame_count=self.frame_count,
                           index_offset=index_offset)
        self._fh.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class DepthContainerReader:
    """Reader for one camera's depth container (see DepthContainerWriter)."""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        header = self._fh.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            self._fh.close()
            raise ValueError(f"{self.path}: truncated depth container header")
        (magic, version, height, width, codec, min_depth, max_depth,
         _reserved, frame_count, index_offset, _pad) = _HEADER.unpack(header)
        if magic != MAGIC:
            self._fh.close()
            raise ValueError(
                f"{self.path}: not a depth container (magic {magic!r})"
            )
        if version != VERSION:
            self._fh.close()
            raise ValueError(
                f"{self.path}: depth container version {version}, "
                f"expected {VERSION}"
            )
        if codec != CODEC_LOG_UINT8:
            self._fh.close()
            raise ValueError(f"{self.path}: unknown depth codec {codec}")
        self.height = int(height)
        self.width = int(width)
        self.clip = (float(min_depth), float(max_depth))
        self._codec = LogDepthToUint8Transform(min_depth_m=self.clip[0],
                                               max_depth_m=self.clip[1])
        self._offsets = self._load_offsets(int(frame_count), int(index_offset))

    def _load_offsets(self, frame_count: int, index_offset: int) -> np.ndarray:
        """The record offsets: from the index when it is complete, else by
        scanning the record framing (a writer that died before close(), or a
        container whose tail was truncated)."""
        if index_offset:
            self._fh.seek(index_offset)
            raw = self._fh.read((frame_count + 1) * 8)
            if len(raw) == (frame_count + 1) * 8:
                return np.frombuffer(raw, "<u8")
        return self._scan_offsets()

    def _scan_offsets(self) -> np.ndarray:
        self._fh.seek(0, os.SEEK_END)
        size = self._fh.tell()
        offsets = [HEADER_SIZE]
        pos = HEADER_SIZE
        while pos + _LENGTH.size <= size:
            self._fh.seek(pos)
            (length,) = _LENGTH.unpack(self._fh.read(_LENGTH.size))
            if length == 0 or pos + _LENGTH.size + length > size:
                break
            pos += _LENGTH.size + length
            offsets.append(pos)
        return np.asarray(offsets, dtype="<u8")

    @property
    def frame_count(self) -> int:
        return len(self._offsets) - 1

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def read(self, i: int) -> np.ndarray:
        """Frame i as (H, W) float32 metres."""
        n = self.frame_count
        if not 0 <= i < n:
            raise IndexError(f"{self.path}: frame {i} out of range (0..{n - 1})")
        start, end = int(self._offsets[i]), int(self._offsets[i + 1])
        self._fh.seek(start)
        (length,) = _LENGTH.unpack(self._fh.read(_LENGTH.size))
        payload = self._fh.read(length)
        if len(payload) != length:
            raise ValueError(f"{self.path}: truncated record {i}")
        arr = np.frombuffer(lz4.frame.decompress(payload), dtype=np.uint8)
        expected = self.height * self.width
        if arr.size != expected:
            raise ValueError(
                f"{self.path}: frame {i} decoded to {arr.size} values, "
                f"expected {expected}"
            )
        return self._codec.decode(arr.reshape(self.height, self.width))

    def read_range(self, a: int = 0, b: Optional[int] = None) -> np.ndarray:
        """Frames [a, b) stacked to (T, H, W) float32 metres."""
        if b is None:
            b = self.frame_count
        if not b > a:
            return np.empty((0, self.height, self.width), dtype=np.float32)
        return np.stack([self.read(i) for i in range(a, b)], axis=0)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class DepthPoseWriter:
    """Writer of a whole (subtask, camera) depth_pose folder."""

    def __init__(self, out_dir, height: int, width: int,
                 clip: tuple[float, float] = (MIN_DEPTH, MAX_DEPTH)):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._container = DepthContainerWriter(self.out_dir / DEPTH_FILE,
                                               height, width, clip)
        self._frame_indices: list[int] = []
        self._extrinsics: list[np.ndarray] = []
        self._intrinsics: list[np.ndarray] = []
        self._closed = False

    def add(self, frame_index: int, depth_m: np.ndarray,
            extrinsics: np.ndarray, intrinsics: np.ndarray) -> None:
        self._container.append(depth_m)
        self._frame_indices.append(int(frame_index))
        self._extrinsics.append(np.asarray(extrinsics, dtype=np.float32))
        self._intrinsics.append(np.asarray(intrinsics, dtype=np.float32))

    def close(self) -> None:
        if self._closed:
            return
        self._container.close()
        write_poses(
            self.out_dir,
            np.asarray(self._frame_indices, dtype=np.int64),
            np.asarray(self._extrinsics, dtype=np.float32),
            np.asarray(self._intrinsics, dtype=np.float32),
            self._container.shape,
        )
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_poses(out_dir, frame_indices: np.ndarray, extrinsics: np.ndarray,
                intrinsics: np.ndarray, shape) -> Path:
    """Write one camera's poses.npz (uncompressed: it is tiny, and this keeps
    the arrays mmap-able)."""
    path = Path(out_dir) / POSES_FILE
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame_indices=np.asarray(frame_indices, dtype=np.int64),
        extrinsics=np.asarray(extrinsics, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        shape=np.asarray(shape, dtype=np.int64),
    )
    return path


class DepthPoseReader:
    """Reader of one (subtask, camera) depth_pose folder.

    The pose arrays are loaded eagerly (they are kilobytes); the depth
    container is opened lazily, so pose-only callers never touch it.
    """

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        with np.load(self.dir / POSES_FILE) as data:
            self.frame_indices = data["frame_indices"].astype(np.int64)
            self.extrinsics = data["extrinsics"].astype(np.float32)
            self.intrinsics = data["intrinsics"].astype(np.float32)
            self.shape = tuple(int(v) for v in data["shape"])
        self._container: Optional[DepthContainerReader] = None

    @staticmethod
    def is_complete(out_dir) -> bool:
        """True when the folder holds a finalized container + poses (the
        marker every producer's --skip-done and the visualizers rely on).

        A staticmethod on purpose: a reader can only be constructed once
        poses.npz exists, so callers must be able to ask this about a
        directory that may only have a pass-1 container.
        """
        out_dir = Path(out_dir)
        return (out_dir / POSES_FILE).is_file() and (out_dir / DEPTH_FILE).is_file()

    @property
    def container(self) -> DepthContainerReader:
        if self._container is None:
            self._container = DepthContainerReader(self.dir / DEPTH_FILE)
        return self._container

    @property
    def depth_count(self) -> int:
        return self.container.frame_count

    def index_of(self, frame_index: int) -> int:
        """Position of an absolute dataset frame index in the container."""
        pos = int(np.searchsorted(self.frame_indices, frame_index))
        if pos >= len(self.frame_indices) or \
                int(self.frame_indices[pos]) != int(frame_index):
            raise KeyError(
                f"{self.dir}: frame {frame_index} not in "
                f"{self.frame_indices[:3]}..{self.frame_indices[-3:]}"
            )
        return pos

    def depth(self, i: int) -> np.ndarray:
        """Depth of container position i, (H, W) float32 metres."""
        return self.container.read(i)

    def depth_at(self, frame_index: int) -> np.ndarray:
        return self.container.read(self.index_of(frame_index))

    def depth_range(self, a: int, b: Optional[int] = None) -> np.ndarray:
        return self.container.read_range(a, b)

    def pose_at(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        """(extrinsics (3, 4), intrinsics (3, 3)) of one absolute frame."""
        i = self.index_of(frame_index)
        return self.extrinsics[i], self.intrinsics[i]

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
