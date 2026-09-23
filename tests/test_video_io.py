"""CPU-only tests for the ffmpeg-pipe frame reader and the video writers.

The reader exists because the local OpenCV build cannot decode AV1 — it opens
the file but ``read()`` returns False — so frames travel through an ffmpeg
subprocess pipe rather than cv2.

The two writers share that pipe; they differ only in the encoder arguments.
``FFV1VideoWriter`` is lossless and ``H264VideoWriter`` is not, so the two
round trips are asserted differently.

The FFV1 round-trip frames are high-entropy noise on purpose: a smooth
gradient survives a lossy codec, so a gradient round trip would still pass if
the encoder silently fell back to a yuv pixel format and quietly made the
"lossless" claim false.  A 32x24x6 video keeps the whole file small enough to
encode in a few milliseconds.
"""

import inspect
import shutil
import subprocess

import numpy as np
import pytest

from utils.file_io.video_io import (
    FFV1VideoWriter,
    H264VideoWriter,
    get_video_info,
    iter_video_frames,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg binary is required"
)

H, W, FPS = 24, 32, 10.0


def _random_frames(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (count, H, W, 3), dtype=np.uint8)


def _write_video(path, frames: np.ndarray, threads: int | None = None) -> None:
    with FFV1VideoWriter(str(path), FPS, (W, H), threads=threads) as writer:
        for frame in frames:
            writer.write_frame(frame)


def _decode_all(path, **kwargs) -> np.ndarray:
    return np.stack(list(iter_video_frames(str(path), **kwargs)))


# -- lossless round trip -----------------------------------------------------


def test_write_then_read_round_trip_is_lossless(tmp_path):
    """Decoded frames must be bit-identical to the frames handed to the writer."""
    frames = _random_frames(6)
    path = tmp_path / "round_trip.mkv"
    _write_video(path, frames)

    decoded = _decode_all(path)

    assert decoded.shape == (6, H, W, 3)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, frames)


def test_round_trip_through_extensionless_temp_name(tmp_path):
    """The container is passed as ``-f matroska`` because callers write to temp
    file names with no ``.mkv`` suffix; nothing may be inferred from the name."""
    frames = _random_frames(4, seed=3)
    path = tmp_path / "tmpy4x9zq1"
    _write_video(path, frames)

    assert np.array_equal(_decode_all(path), frames)


def test_frame_count_tracks_written_frames(tmp_path):
    """``frame_count`` is the caller's only cheap check that every frame landed."""
    frames = _random_frames(5, seed=4)
    writer = FFV1VideoWriter(str(tmp_path / "five.mkv"), FPS, (W, H))

    assert writer.frame_count == 0
    for frame in frames:
        writer.write_frame(frame)
    assert writer.frame_count == len(frames)

    writer.close()
    # An odd frame count leaves a short final block for the encoder's slices.
    assert np.array_equal(_decode_all(tmp_path / "five.mkv"), frames)


# -- subset reads ------------------------------------------------------------


def test_iter_video_frames_indices_match_full_decode(tmp_path):
    """Indices select frames off the full decode, in the order they are given."""
    frames = _random_frames(6, seed=1)
    path = tmp_path / "subset.mkv"
    _write_video(path, frames)
    all_frames = _decode_all(path)

    indices = [1, 3, 5]
    picked = list(iter_video_frames(str(path), indices=indices))

    assert len(picked) == len(indices)
    for got, idx in zip(picked, indices):
        assert np.array_equal(got, all_frames[idx])


def test_iter_video_frames_accepts_an_explicit_size(tmp_path):
    """A caller that already probed the video can skip our own ffprobe call."""
    frames = _random_frames(6, seed=2)
    path = tmp_path / "sized.mkv"
    _write_video(path, frames)

    picked = list(iter_video_frames(str(path), indices=[0, 5], size=(W, H)))

    assert len(picked) == 2
    assert np.array_equal(picked[0], frames[0])
    assert np.array_equal(picked[1], frames[5])


def test_iter_video_frames_does_not_pad_a_variable_frame_rate_source(tmp_path):
    """Frames with uneven timestamps must come back one for one.

    ffmpeg's default frame sync is CFR: it pads a variable-frame-rate source
    up to its nominal rate by duplicating frames, which turns this 6-frame
    clip into 27 frames of "every frame in order".
    """
    frames = _random_frames(6, seed=10)
    path = tmp_path / "vfr_src.mkv"
    _write_video(path, frames)

    vfr = tmp_path / "vfr.mkv"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
            # Push one frame's timestamp far out without dropping any.
            "-vf", "setpts='if(eq(N\\,3),PTS*8,PTS)'",
            "-fps_mode", "passthrough",
            "-c:v", "ffv1", "-pix_fmt", "bgr0", "-f", "matroska", str(vfr),
        ],
        check=True, capture_output=True,
    )

    decoded = list(iter_video_frames(str(vfr), size=(W, H)))

    assert len(decoded) == len(frames)
    assert np.array_equal(np.stack(decoded), frames)


def test_iter_video_frames_reemits_out_of_order_requests(tmp_path):
    """ffmpeg can only hand frames back in decode order, so anything else has
    to be re-emitted in the order the caller asked for."""
    frames = _random_frames(6, seed=8)
    path = tmp_path / "out_of_order.mkv"
    _write_video(path, frames)

    picked = list(iter_video_frames(str(path), indices=[5, 2, 2, 0]))

    assert len(picked) == 4
    for got, idx in zip(picked, [5, 2, 2, 0]):
        assert np.array_equal(got, frames[idx])


def test_iter_video_frames_with_no_indices_yields_nothing(tmp_path):
    """An empty index list must not degrade to "decode everything"."""
    frames = _random_frames(6, seed=9)
    path = tmp_path / "none_wanted.mkv"
    _write_video(path, frames)

    assert list(iter_video_frames(str(path), indices=[])) == []


def test_iter_video_frames_is_lazy(tmp_path):
    """Frames must be pulled as the caller iterates — the exports this feeds
    are encoded frame by frame, so materialising a whole AV1 video up front
    would defeat the point of the pipe."""
    frames = _random_frames(6, seed=5)
    path = tmp_path / "lazy.mkv"
    _write_video(path, frames)

    gen = iter_video_frames(str(path))
    assert inspect.isgenerator(gen)

    # Abandoning a half-read generator must not leave an ffmpeg process behind.
    first = next(gen)
    assert np.array_equal(first, frames[0])
    gen.close()


# -- errors ------------------------------------------------------------------


def test_iter_video_frames_raises_ffmpeg_error(tmp_path):
    """A failed ffmpeg must surface its stderr, not an empty frame stream."""
    missing = tmp_path / "does_not_exist.mkv"

    with pytest.raises(RuntimeError, match="ffmpeg"):
        list(iter_video_frames(str(missing), size=(W, H)))


def test_iter_video_frames_rejects_an_unreadable_file(tmp_path):
    """ffmpeg's non-zero exit must become a RuntimeError, not an empty stream."""
    garbage = tmp_path / "garbage.mkv"
    garbage.write_bytes(b"not a video at all" * 64)

    with pytest.raises(RuntimeError, match="ffmpeg"):
        list(iter_video_frames(str(garbage), size=(W, H)))


def test_iter_video_frames_rejects_a_ragged_final_frame(tmp_path):
    """A byte stream that stops part-way through a frame must raise rather
    than hand back a short array.

    Declaring a height one row taller than the file is what a truncated read
    at EOF looks like to the reader: the frame boundaries no longer divide
    the stream evenly and the last pull comes up short.
    """
    frames = _random_frames(6, seed=11)
    path = tmp_path / "ragged.mkv"
    _write_video(path, frames)

    with pytest.raises(RuntimeError, match="ended mid-frame"):
        list(iter_video_frames(str(path), size=(W, H + 1)))


def test_truncated_video_yields_only_whole_frames(tmp_path):
    """A damaged tail must never be resliced into a short frame.

    ffmpeg's Matroska demuxer drops an incomplete trailing frame outright
    (verified against this build), so what this pins is the contract that
    actually holds: whatever survives decoding has the declared shape and the
    exact source pixels, and the frame boundaries never shift.
    """
    frames = _random_frames(4, seed=6)
    path = tmp_path / "truncated.mkv"
    _write_video(path, frames)

    damaged = tmp_path / "damaged.mkv"
    damaged.write_bytes(path.read_bytes()[:-400])

    decoded = list(iter_video_frames(str(damaged), size=(W, H)))

    assert 0 < len(decoded) <= len(frames)
    for got, expected in zip(decoded, frames):
        assert got.shape == (H, W, 3)
        assert np.array_equal(got, expected)


def test_write_frame_rejects_dtype_and_shape_mismatch(tmp_path):
    """Validation happens on the first frame: a mismatch would otherwise be
    written into the rawvideo pipe as silent corruption."""
    writer = FFV1VideoWriter(str(tmp_path / "bad.mkv"), FPS, (W, H))
    try:
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W, 4), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H + 1, W, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W + 1, 3), dtype=np.uint8))

        assert writer.frame_count == 0
        # A rejected frame must not poison the writer for later good frames.
        writer.write_frame(np.zeros((H, W, 3), dtype=np.uint8))
        assert writer.frame_count == 1
    finally:
        writer.close()


# -- lifecycle ---------------------------------------------------------------


def test_close_is_idempotent(tmp_path):
    """close() is safe to call twice — callers finalise in a finally block."""
    frames = _random_frames(3, seed=7)
    path = tmp_path / "twice.mkv"
    writer = FFV1VideoWriter(str(path), FPS, (W, H))
    for frame in frames:
        writer.write_frame(frame)

    assert not writer.closed
    writer.close()
    assert writer.closed
    writer.close()
    assert writer.closed

    assert np.array_equal(_decode_all(path), frames)


# -- H.264 (libx264) mp4 writer ---------------------------------------------
#
# Same pipe, different encoder arguments: OpenCV itself has no H.264 encoder
# in this build and its only working fallback is MPEG-4 Part 2, which is far
# too big to write a video whose whole job is to be watched.

# Mean absolute difference allowed across a yuv420p H.264 round trip of the
# structured frames below; measured at ~4, so there is room for encoder
# variation while a channel swap or a wrong frame still fails loudly.
_H264_MAD_TOLERANCE = 12.0


def _structured_frames(count: int) -> np.ndarray:
    """Gradients plus a hard-edged block — content a lossy codec approximates
    closely, unlike the per-pixel noise the FFV1 round trip needs."""
    yy, xx = np.mgrid[0:H, 0:W]
    frames = np.empty((count, H, W, 3), np.uint8)
    for i in range(count):
        frames[i, ..., 0] = (xx * 255) // (W - 1)
        frames[i, ..., 1] = (yy * 255) // (H - 1)
        frames[i, ..., 2] = (i * 40) % 256
        frames[i, 4:12, 4:12] = 250
    return frames


def test_h264_round_trip_is_close_but_not_lossless(tmp_path):
    """This writer is for looking at, not for data.

    ``yuv420p`` is what makes the result playable everywhere and RGB -> YUV ->
    RGB is not a round trip, so the decoded frames only have to be CLOSE to
    the source — never bit-identical, hence a mean-absolute-difference bound
    rather than the FFV1 tests' ``np.array_equal``.
    """
    frames = _structured_frames(6)
    path = tmp_path / "close.mp4"
    with H264VideoWriter(str(path), FPS, (W, H)) as writer:
        for frame in frames:
            writer.write_frame(frame)

    decoded = _decode_all(path, size=(W, H))

    assert decoded.shape == (6, H, W, 3)
    assert decoded.dtype == np.uint8
    difference = np.abs(decoded.astype(np.int16) - frames.astype(np.int16))
    assert difference.mean() < _H264_MAD_TOLERANCE


def test_h264_writes_the_size_and_frame_count_it_was_asked_for(tmp_path):
    """Neither dimension may drift: the caller sizes its frames from ``size``,
    and a video that comes back a different size is unusable in a viewer."""
    frames = _structured_frames(5)
    path = tmp_path / "sized.mp4"
    writer = H264VideoWriter(str(path), FPS, (W, H))

    assert writer.frame_count == 0
    for frame in frames:
        writer.write_frame(frame)
    assert writer.frame_count == len(frames)
    writer.close()

    assert get_video_info(str(path)) == (len(frames), pytest.approx(FPS), W, H)
    assert _decode_all(path, size=(W, H)).shape == (len(frames), H, W, 3)


def test_h264_higher_crf_makes_a_smaller_file(tmp_path):
    """crf is the size/quality dial, and small files are the entire reason
    this writer exists — OpenCV's only working encoder here (``mp4v``,
    MPEG-4 Part 2) makes files several times larger for the same look.

    High-entropy noise, at a size where the encoder has real work to do, so
    the two settings actually differ (~35 kB vs ~2 kB, measured).
    """
    side, count = 64, 8
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 256, (count, side, side, 3), dtype=np.uint8)

    sizes = {}
    for crf in (10, 45):
        path = tmp_path / f"crf{crf}.mp4"
        with H264VideoWriter(str(path), FPS, (side, side), crf=crf) as writer:
            for frame in frames:
                writer.write_frame(frame)
        sizes[crf] = path.stat().st_size

    assert 0 < sizes[45] < sizes[10]


def test_h264_close_is_idempotent(tmp_path):
    """close() is safe to call twice — callers finalise in a finally block."""
    frames = _structured_frames(3)
    path = tmp_path / "twice.mp4"
    writer = H264VideoWriter(str(path), FPS, (W, H))
    for frame in frames:
        writer.write_frame(frame)

    assert not writer.closed
    writer.close()
    assert writer.closed
    writer.close()
    assert writer.closed

    assert _decode_all(path, size=(W, H)).shape == (3, H, W, 3)


# -- H.264 errors ------------------------------------------------------------


def test_h264_rejects_an_odd_width(tmp_path):
    """yuv420p subsamples chroma 2x2, so an odd dimension cannot be encoded;
    the error must name it rather than let ffmpeg fail obscurely or round it
    off behind the caller's back."""
    with pytest.raises(ValueError, match="even width"):
        H264VideoWriter(str(tmp_path / "odd_width.mp4"), FPS, (W + 1, H))


def test_h264_rejects_an_odd_height(tmp_path):
    with pytest.raises(ValueError, match="even height"):
        H264VideoWriter(str(tmp_path / "odd_height.mp4"), FPS, (W, H + 1))


def test_h264_write_frame_rejects_dtype_and_shape_mismatch(tmp_path):
    """Validation happens on the first frame: a mismatch would otherwise be
    written into the rawvideo pipe as silent corruption."""
    writer = H264VideoWriter(str(tmp_path / "bad.mp4"), FPS, (W, H))
    try:
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W, 4), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H + 2, W, 3), dtype=np.uint8))
        with pytest.raises(ValueError):
            writer.write_frame(np.zeros((H, W + 2, 3), dtype=np.uint8))

        assert writer.frame_count == 0
        # A rejected frame must not poison the writer for later good frames.
        writer.write_frame(np.zeros((H, W, 3), dtype=np.uint8))
        assert writer.frame_count == 1
    finally:
        writer.close()
