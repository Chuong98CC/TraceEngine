
import json
import os
import subprocess

import cv2
import numpy as np


def _parse_ffprobe_fraction(frac_str: str) -> float:
    """Parse an ffprobe frame-rate string like ``\"30000/1001\"`` → float."""
    parts = frac_str.split("/")
    num = float(parts[0])
    den = float(parts[1]) if len(parts) > 1 else 1.0
    return num / den if den != 0 else 0.0

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