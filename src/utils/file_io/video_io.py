
import json
import os
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from typing import Self

import cv2
import numpy as np

# Cap on a single raw frame chunk written into an ffmpeg pipe.
_RAW_CHUNK_BYTES = 1 << 20


def _parse_ffprobe_fraction(frac_str: str) -> float:
    """Parse an ffprobe frame-rate string like ``\"30000/1001\"`` → float."""
    parts = frac_str.split("/")
    num = float(parts[0])
    den = float(parts[1]) if len(parts) > 1 else 1.0
    return num / den if den != 0 else 0.0

def _stderr_text(sink) -> str:
    """Read a captured stderr sink (a temp file) back as a single string."""
    sink.seek(0)
    return sink.read().decode("utf-8", errors="replace").strip()

def _read_exactly(stream, n: int) -> bytes:
    """Read *n* bytes from a pipe, coming up short only at EOF.

    A short ``read()`` on a pipe means "nothing buffered yet", not "end of
    stream", so one call can hand back a partial frame; only an empty read is
    EOF.  Guessing wrong would slice a frame at the wrong offset and shift
    every following pixel.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)

def get_video_info(video_path: str) -> tuple[int, float, int, int]:
    """Probe a video file for metadata using ffprobe (fast, no decode).

    Falls back to ``cv2.VideoCapture`` when ffprobe is unavailable.

    Returns
    -------
    tuple[int, float, int, int]
        ``(total_frames, fps, width, height)``.  *total_frames* is 0 when
        the container cannot report it.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v:0",
                video_path,
            ],
            capture_output=True, text=True, timeout=15,
            check=True,
        )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise ValueError("no video stream found")

        s = streams[0]
        width = int(s["width"])
        height = int(s["height"])
        fps = _parse_ffprobe_fraction(s.get("r_frame_rate", "30/1"))

        nb_frames = s.get("nb_frames")
        if nb_frames is not None and nb_frames != "N/A":
            total_frames = int(nb_frames)
        else:
            duration = float(s.get("duration", 0))
            total_frames = int(round(duration * fps)) if duration > 0 else 0

        if fps <= 0:
            fps = 30.0

        return total_frames, fps, width, height

    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError,
            KeyError, ValueError):
        pass  # fall through to cv2 fallback

    # -- OpenCV fallback ------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Verify the codec is actually decodable by reading a test frame.
    ret, _ = cap.read()
    cap.release()

    if not ret:
        raise IOError(
            f"Cannot decode frames from: {video_path}\n"
            "The video codec may not be supported by your OpenCV/ffmpeg build. "
            "Try re-encoding to H.264:\n"
            "  ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 23 output.mp4"
        )

    if fps <= 0:
        fps = 30.0

    return total_frames, fps, width, height

def open_video(video_path: str):
    """Open a video file and return (cap, fps, total_frames, width, height)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, total_frames, width, height

def iter_video_frames(
    video_path: str,
    indices: Sequence[int] | None = None,
    size: tuple[int, int] | None = None,
) -> Iterator[np.ndarray]:
    """Yield ``(H, W, 3)`` uint8 RGB frames decoded through an ffmpeg pipe.

    Decoding goes through ffmpeg rather than ``cv2.VideoCapture`` because the
    local OpenCV build opens AV1 files but cannot decode them (``read()``
    returns False), while ffmpeg handles every codec it was built against.

    Parameters
    ----------
    video_path : str
        Input video; ffmpeg auto-detects the container.
    indices : Sequence[int] | None
        ``None`` yields every frame in order.  A sequence of 0-based frame
        indices yields only those frames — an ffmpeg ``select`` filter drops
        the rest as they are decoded, so asking for a handful of frames never
        pulls a whole video through Python.  Frames come off the decode in
        ascending order; an out-of-order or repeated request is re-emitted in
        the order given, buffering only the frames asked for.  Indices past
        the end of the video are skipped.
    size : tuple[int, int] | None
        ``(width, height)``; probed with :func:`get_video_info` when ``None``.

    Yields
    ------
    np.ndarray
        ``(H, W, 3)`` uint8 RGB, writable so a caller can draw on a frame.

    Raises
    ------
    RuntimeError
        If ffmpeg exits non-zero or its output stops mid-frame, with ffmpeg's
        own stderr in the message.
    """
    if size is None:
        _, _, width, height = get_video_info(video_path)
    else:
        width, height = size
    frame_bytes = width * height * 3

    wanted = list(indices) if indices is not None else None
    if wanted is not None and not wanted:
        return  # an empty select expression would be read as "keep everything"
    decode_order = sorted(set(wanted)) if wanted is not None else []

    cmd = ["ffmpeg", "-loglevel", "error", "-i", video_path]
    if wanted is not None:
        select = "+".join(f"eq(n\\,{i})" for i in decode_order)
        cmd += ["-vf", f"select='{select}'"]
    # -fps_mode passthrough means "hand the frames over as they are".  It is
    # an *output* option, so it has to sit after -i and after -vf.  The
    # default it replaces, CFR sync, pads the output up to the input's
    # nominal rate: it duplicates frames to fill the holes the select filter
    # leaves behind, and on a variable-frame-rate source it turns a 6-frame
    # clip into 27 frames.
    cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    # stdin is /dev/null: an inherited stdin lets ffmpeg grab it for
    # interactive key handling and stall when the parent never writes.
    # stderr goes to a temp file rather than a pipe — a decoder erroring on
    # every frame of a broken file would otherwise fill the 64 KiB pipe
    # buffer and wedge ffmpeg while we are still reading stdout.
    stderr_sink = tempfile.TemporaryFile()
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=stderr_sink,
    )

    def _decoded() -> Iterator[np.ndarray]:
        while True:
            chunk = _read_exactly(process.stdout, frame_bytes)
            if not chunk:
                break
            if len(chunk) != frame_bytes:
                raise RuntimeError(
                    f"ffmpeg ended mid-frame on {video_path}: {len(chunk)} of "
                    f"{frame_bytes} bytes for a {width}x{height} frame"
                )
            frame = np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
            yield frame.copy()  # writable; frombuffer() hands back a read-only view

    try:
        if wanted is None or wanted == decode_order:
            yield from _decoded()
        else:
            buffered = dict(zip(decode_order, _decoded()))
            for idx in wanted:
                if idx in buffered:
                    yield buffered[idx]

        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {returncode}) reading {video_path}: "
                f"{_stderr_text(stderr_sink)}"
            )
    finally:
        # A consumer that abandons the generator mid-decode must not leave an
        # encoder process running behind it.
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        stderr_sink.close()

class VideoWriter:
    """Streaming video writer — writes frames one at a time via
    :meth:`write_frame` or :meth:`write_overlay_frame`.

    The output size is inferred from the first frame written when ``size``
    is not given at construction.  Call :meth:`close` (or use as a context
    manager) to finalise the file.

    Parameters
    ----------
    output_path : str
        Output ``.mp4`` file path.
    fps : float
        Frame rate for the output video.
    size : tuple[int, int] | None
        ``(width, height)``.  Inferred from the first frame when ``None``.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        size: tuple[int, int] | None = None,
    ) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._path = output_path
        self._fps = fps
        self._size = size
        self._writer: cv2.VideoWriter | None = None
        self._frame_count = 0

    # -- lazy init ----------------------------------------------------------
    def _ensure_writer(self, w: int, h: int) -> None:
        if self._writer is not None:
            return
        if self._size is None:
            self._size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, self._size)

    # -- write methods ------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single BGR frame (H×W×3 uint8)."""
        h, w = frame.shape[:2]
        self._ensure_writer(w, h)
        self._writer.write(frame)
        self._frame_count += 1

    def write_overlay_frame(
        self,
        original: np.ndarray,
        overlay: np.ndarray,
        alpha: float = 0.5,
    ) -> None:
        """Blend *overlay* onto *original* and write the result.

        Parameters
        ----------
        original : np.ndarray
            Base BGR image (H×W×3 uint8).
        overlay : np.ndarray
            Overlay BGR image, same size as *original*.
        alpha : float
            Blend weight for the overlay (0 = pure original, 1 = pure overlay).
        """
        blended = cv2.addWeighted(original, 1.0 - alpha, overlay, alpha, 0)
        self.write_frame(blended)

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Release the underlying encoder and finalise the file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(
                f"Output video saved: {self._path} "
                f"({self._frame_count} frames)"
            )

    @property
    def closed(self) -> bool:
        return self._writer is None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()

class _FFmpegVideoWriter:
    """Pipe machinery shared by the ffmpeg-backed video writers.

    Frames go in one at a time as raw RGB on the ffmpeg subprocess's stdin
    and come out encoded by whatever output arguments the subclass supplies.
    Everything codec-independent lives here — the pipe setup, the chunked
    write, the frame validation, ``close()``, ``frame_count``, ``closed``
    and the context manager.

    Parameters
    ----------
    output_path : str
        Output path.  The container is forced by the subclass's ``-f``, so
        the name needs no suffix and may be a temporary file.
    fps : float
        Frame rate recorded for the stream.
    size : tuple[int, int]
        ``(width, height)``.  Required up front — the rawvideo demuxer needs
        the geometry before the first byte arrives, so unlike
        :class:`VideoWriter` the size cannot be inferred from a frame.
    encoder_args : Sequence[str]
        Output-side ffmpeg arguments: the encoder, its options and the
        explicit container ``-f``, followed (by this class) by the path.
    label : str
        Class name used in error messages, so a failure names the writer the
        caller actually constructed.
    threads : int | None
        Encoder thread count; ffmpeg's own default when ``None``.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        size: tuple[int, int],
        *,
        encoder_args: Sequence[str],
        label: str,
        threads: int | None = None,
    ) -> None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        width, height = int(size[0]), int(size[1])
        self._label = label
        # Before ffmpeg starts, so a rejected geometry cannot leak a process.
        self._validate_size(width, height)
        self._path = output_path
        self._width = width
        self._height = height
        self._frame_count = 0
        self._stderr = tempfile.TemporaryFile()

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
        ]
        if threads is not None:
            cmd += ["-threads", str(threads)]
        cmd += list(encoder_args)
        cmd += [output_path]
        # stderr goes to a temp file, not a pipe: an encoder that errors while
        # we are writing frames would fill a 64 KiB stderr pipe and then stop
        # draining stdin, deadlocking the write below.
        self._process: subprocess.Popen | None = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=self._stderr
        )

    def _validate_size(self, width: int, height: int) -> None:
        """Reject geometry the encoder cannot represent.

        A no-op for encoders that take any size; overridden by subclasses
        whose pixel format imposes constraints, to raise a clear error naming
        the offending dimension rather than let ffmpeg fail obscurely.
        """

    # -- write methods ------------------------------------------------------

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single ``(H, W, 3)`` uint8 RGB frame.

        The frame is validated before it reaches the pipe — a mismatch would
        otherwise be encoded as silent corruption.
        """
        if self._process is None:
            raise RuntimeError(f"{self._label} is already closed: {self._path}")
        expected = (self._height, self._width, 3)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape != expected:
            raise ValueError(
                f"{self._label} expects a {expected} uint8 RGB frame, got "
                f"{frame.shape} of {frame.dtype}"
            )
        # A memoryview onto the frame's own buffer, so a frame is not copied
        # before it goes into the pipe; slicing it keeps any single write
        # small even for 4K frames.
        raw = np.ascontiguousarray(frame).data.cast("B")
        try:
            for start in range(0, len(raw), _RAW_CHUNK_BYTES):
                self._process.stdin.write(raw[start : start + _RAW_CHUNK_BYTES])
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"ffmpeg exited before all frames were written to {self._path}: "
                f"{_stderr_text(self._stderr)}"
            ) from exc
        self._frame_count += 1

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """Flush the pipe, wait for ffmpeg and finalise the file.

        Safe to call more than once, like :meth:`VideoWriter.close`; nothing
        is flushed on the second call.  Raises if ffmpeg reported a failure.
        """
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg died first; the exit code below carries the reason
        returncode = process.wait()
        stderr = _stderr_text(self._stderr)
        self._stderr.close()
        if returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {returncode}) writing {self._path}: {stderr}"
            )
        print(f"Output video saved: {self._path} ({self._frame_count} frames)")

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def closed(self) -> bool:
        return self._process is None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class FFV1VideoWriter(_FFmpegVideoWriter):
    """Lossless RGB video writer — frames go in one at a time over a rawvideo
    pipe to ffmpeg, which encodes them as FFV1 in a Matroska container.

    Use this where :class:`VideoWriter` would silently rewrite the pixels: the
    ffv1 encoder is driven with ``-pix_fmt bgr0``, a packed byte layout it
    stores verbatim, whereas the lossy yuv formats are not an RGB round trip.

    Parameters
    ----------
    output_path : str
        Output path.  The container is forced with ``-f matroska``, so the
        name needs no ``.mkv`` suffix and may be a temporary file.
    fps : float
        Frame rate recorded for the stream.
    size : tuple[int, int]
        ``(width, height)``.  Required up front — the rawvideo demuxer needs
        the geometry before the first byte arrives, so unlike
        :class:`VideoWriter` the size cannot be inferred from a frame.
    threads : int | None
        Encoder thread count; ffmpeg's own default when ``None``.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        size: tuple[int, int],
        threads: int | None = None,
    ) -> None:
        super().__init__(
            output_path,
            fps,
            size,
            threads=threads,
            label="FFV1VideoWriter",
            encoder_args=[
                "-c:v", "ffv1",
                "-level", "3",
                # bgr0 is the point of this argument: any yuv pixel format
                # would make the "lossless" claim false, because RGB -> YUV ->
                # RGB is not a round trip.
                "-pix_fmt", "bgr0",
                "-slices", "4",
                "-slicecrc", "1",
                # Explicit so a bare temp file name still lands in Matroska;
                # ffmpeg would otherwise refuse to guess from the extension.
                "-f", "matroska",
            ],
        )


class H264VideoWriter(_FFmpegVideoWriter):
    """H.264 (libx264) mp4 writer for small visualization videos.

    Use this for anything whose only job is to be watched: the OpenCV build
    here has no H.264 encoder at all (``avc1``/``H264``/``h264`` all fail with
    "Could not find encoder for codec_id=27") and its only working fallback,
    MPEG-4 Part 2 (``mp4v``), makes files several times larger for the same
    look.  libx264 is reached over the ffmpeg pipe instead.

    This writer is deliberately lossy — ``-pix_fmt yuv420p`` is what makes the
    result playable everywhere, and RGB -> YUV -> RGB is not a round trip, so
    use :class:`FFV1VideoWriter` when the pixels have to survive exactly.

    Parameters
    ----------
    output_path : str
        Output path.  The container is forced with ``-f mp4``, so the name
        needs no ``.mp4`` suffix and may be a temporary file.
    fps : float
        Frame rate recorded for the stream.
    size : tuple[int, int]
        ``(width, height)``; both must be **even**, since ``yuv420p``
        subsamples chroma 2x2.  An odd dimension raises rather than being
        silently rounded or padded — a video that comes back a different size
        than the caller asked for is a worse failure than an exception.
    threads : int | None
        Encoder thread count; ffmpeg's own default when ``None``.
    crf : int
        Constant Rate Factor — the size/quality dial, lower is better and
        bigger.  23 is x264's default and a reasonable "small for
        visualization" point: 0 is lossless and huge, the high 20s to low 30s
        make visibly softer but much smaller files.
    """

    def __init__(
        self,
        output_path: str,
        fps: float,
        size: tuple[int, int],
        threads: int | None = None,
        crf: int = 23,
    ) -> None:
        super().__init__(
            output_path,
            fps,
            size,
            threads=threads,
            label="H264VideoWriter",
            encoder_args=[
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                # Puts the moov atom up front, so a browser or a video player
                # can start playing before the whole file has arrived.
                "-movflags", "+faststart",
                "-f", "mp4",
            ],
        )

    def _validate_size(self, width: int, height: int) -> None:
        for name, value in (("width", width), ("height", height)):
            if value % 2:
                raise ValueError(
                    f"{self._label} requires an even {name} for yuv420p chroma "
                    f"subsampling, got {name}={value}"
                )
