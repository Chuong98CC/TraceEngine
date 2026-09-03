import sys
from pathlib import Path
from typing import Optional

import cv2
import lz4.frame
import numpy as np

MAX_DEPTH = 1.501  # Maximum depth in meters for encoding/decoding
MIN_DEPTH = 0.001  # Minimum depth in meters for encoding/decoding
QUANT_MAX = 255  # 8-bit quantization levels of the log-depth codec

def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    depth = depth_map[..., 0] if depth_map.ndim == 4 else depth_map
    N, H, W = depth.shape

    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = np.broadcast_to(x[None], (N, H, W))
    y = np.broadcast_to(y[None], (N, H, W))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1,
    )

    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    return np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)), camera_points - t[:, None, None, :])

def _pixel_to_camera_point(
        uv: tuple[int, int],
        depth_m: np.ndarray,
        K: np.ndarray) -> np.ndarray:
    """Unproject a pixel coordinate to a 3D point in camera space.

    Args:
        uv: (u, v) pixel coordinate (x, y convention).
        depth_m: (H, W) float32 depth map in metres.
        K: (3, 3) camera intrinsic matrix.

    Returns:
        (3,) float64 [X, Y, Z] point in camera coordinates (OpenCV: x-right, y-down, z-forward).
        Returns zeros if the depth at the pixel is invalid.
    """
    u, v = int(uv[0]), int(uv[1])
    H, W = depth_m.shape[:2]

    if not (0 <= u < W and 0 <= v < H):
        raise ValueError(f"Pixel ({u},{v}) out of bounds for depth map ({W}x{H}).")

    # Robust scalar extraction: handles 2D maps and multi-channel arrays
    Z = float(np.asarray(depth_m[v, u]).ravel()[0])
    if Z <= 0.0 or not np.isfinite(Z):
        print(f"Warning: invalid depth ({Z:.4f}) at pixel ({u},{v}), returning zeros.", file=sys.stderr)
        return np.zeros(3, dtype=np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)

def measure_pixel_distance(
    depth_m: np.ndarray,
    K: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
) -> dict:
    """Compute the real-world Euclidean distance between two pixels.

    Unprojects each pixel to a 3D camera-space point using the depth map and
    intrinsics, then returns the Euclidean distance between them.

    Args:
        depth_m: (H, W) float32 depth map in metres (e.g. from
                 ``approximate_depth_from_grayscale`` or a depth sensor).
        K: (3, 3) camera intrinsic matrix with fx/fy (upper-left diagonal)
           and cx/cy (rightmost column, first two rows).
        p1: (u, v) first pixel coordinate.
        p2: (u, v) second pixel coordinate.

    Returns:
        dict with keys:
            p1_3d   — (3,) float64 [X, Y, Z] of first pixel.
            p2_3d   — (3,) float64 [X, Y, Z] of second pixel.
            dist_m  — float, Euclidean distance in metres.
    """
    P1 = _pixel_to_camera_point(p1, depth_m, K)
    P2 = _pixel_to_camera_point(p2, depth_m, K)
    dist_m = float(np.linalg.norm(P2 - P1))
    return {"p1_3d": P1, "p2_3d": P2, "dist_m": dist_m}

def draw_measurement(
    image: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    dist_m: float,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Draw the two measured points and the distance between them on an image.

    Circles at each point, a line connecting them, and a label with the distance
    in metres. Works on BGR, RGB, or single-channel (greyscale / depth) images.

    Args:
        image: (H, W) or (H, W, 3) uint8 image.
        p1, p2: (u, v) pixel coordinates of the two points.
        dist_m: Euclidean distance in metres to display on the label.
        save_path: if given, writes the annotated image to this path.

    Returns:
        (H, W, 3) uint8 BGR copy of the image with the overlay drawn on it.
    """
    # Make a writable BGR copy
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    # Colours (BGR)
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)

    # Draw circles at each point
    radius = max(4, min(out.shape[0], out.shape[1]) // 200)
    cv2.circle(out, p1, radius, green, thickness=-1)
    cv2.circle(out, p2, radius, red, thickness=-1)
    cv2.circle(out, p1, radius + 1, white, thickness=1)
    cv2.circle(out, p2, radius + 1, white, thickness=1)

    # Draw line between points
    cv2.line(out, p1, p2, (255, 255, 0), thickness=2)

    # Labels
    cv2.putText(out, f"P1 {p1}", (p1[0] + 10, p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1)
    cv2.putText(out, f"P2 {p2}", (p2[0] + 10, p2[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1)

    # Distance label at midpoint
    mx = (p1[0] + p2[0]) // 2
    my = (p1[1] + p2[1]) // 2
    label = f"{dist_m:.4f} m"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Draw filled background rectangle so text is legible over any content
    cv2.rectangle(out, (mx - tw // 2 - 4, my - th - 4), (mx + tw // 2 + 4, my + 4), black, -1)
    cv2.putText(out, label, (mx - tw // 2, my + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)

    if save_path is not None:
        cv2.imwrite(save_path, out)

    return out


# ---------------------------------------------------------------------------
# Depth storage units: all .lz4 depth files store float metres log-encoded
# to uint8 over [MIN_DEPTH, MAX_DEPTH] (LogDepthToUint8Transform), lz4-frame
# compressed; pose (extrinsics/intrinsics) lives in a separate .npz per frame.
# ---------------------------------------------------------------------------
def _depth_to_float_metres(depth) -> np.ndarray:
    """Depth input -> float32 metres, units auto-detected.

    uint16 arrays whose max exceeds MAX_DEPTH are raw millimetres (the
    astribot storage convention) and are divided by 1000; float arrays are
    taken as already in metres. Everything else is assumed to be metres.
    """
    arr = np.asarray(depth)
    if arr.dtype == np.uint16 and arr.max() > MAX_DEPTH:
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def save_depth_lz4(depth, path: Path) -> None:
    """Compress a depth map into an .lz4 file readable by load_depth_lz4.

    The depth — float metres, or uint16 mm (auto-detected, see
    LogDepthToUint8Transform.encode) — is log-encoded to uint8 over
    [MIN_DEPTH, MAX_DEPTH] before lz4-frame compression."""
    Path(path).write_bytes(lz4.frame.compress(
        LogDepthToUint8Transform().encode(np.asarray(depth)).tobytes()))


def save_depth_m_lz4(depth_m, path: Path) -> None:
    """Save a float depth map in metres as log-encoded uint8 (.lz4)."""
    save_depth_lz4(depth_m, path)


def load_depth_lz4(depth_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a depth map from an lz4 file (see save_depth_lz4): lz4-frame
    compressed log-encoded uint8, decoded to float32 metres. Accepts str or
    Path (streaming_utils passes os.path.join strs)."""
    raw = Path(depth_path).read_bytes()
    try:
        decoded = lz4.frame.decompress(raw)
    except lz4.frame.LZ4FrameError as exc:
        raise RuntimeError(f"Failed to decompress depth file {depth_path}: {exc}") from exc
    arr = np.frombuffer(decoded, dtype=np.uint8)
    expected = shape[0] * shape[1]
    if arr.size != expected:
        raise ValueError(f"Unexpected decoded depth size for {depth_path}: "
                         f"got {arr.size}, expected {expected}")
    return LogDepthToUint8Transform().decode(arr.reshape(shape))

def normalize_depth_for_display(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        max_value = float(depth.max()) if depth.size else 1.0
        if max_value <= 0:
            max_value = 1.0
        out = (depth.astype(np.float32) / max_value * 255.0).astype(np.uint8)
    elif depth.dtype == np.float32 or depth.dtype == np.float64:
        min_val = float(depth.min()) if depth.size else 0.0
        max_val = float(depth.max()) if depth.size else 1.0
        if max_val - min_val <= 1e-6:
            max_val = min_val + 1.0
        out = ((depth - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    else:
        out = cv2.convertScaleAbs(depth)

    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    return out


class LogDepthToUint8Transform:
    """Deterministic log-space depth codec: metres <-> uint8 [0, 255].

    Depth in metres is log-transformed and quantized to 8 bits over
    ``[min_depth_m, max_depth_m]``; values outside the range clip to the
    range ends and 0 (invalid) is preserved as 0. ``encode``/``decode`` are
    exact inverses up to the 8-bit quantization — the storage format of
    every .lz4 depth file (see save_depth_lz4/load_depth_lz4).

    Args:
        min_depth_m: Minimum clip distance in meters.
        max_depth_m: Maximum clip distance in meters.
        shift_m: Small epsilon added before ``log`` so zero/invalid pixels
            stay finite (they are zeroed afterwards).
    """
    def __init__(
        self,
        min_depth_m: float = MIN_DEPTH,
        max_depth_m: float = MAX_DEPTH,
        shift_m: float = 0.001,
    ):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.shift_m = shift_m

        # Precompute log bounds
        self.z_min = float(np.log(min_depth_m + shift_m))
        self.z_max = float(np.log(max_depth_m + shift_m))

    def encode(self, depth) -> np.ndarray:
        """Depth -> (H, W) uint8 in [0, 255].

        The input is float depth in metres, or uint16 in mm — units are
        auto-detected (see ``_depth_to_float_metres``). Returns 0 for
        zero/invalid pixels.

        Args:
            depth: (H, W) or (1, H, W) depth map (uint16 mm or float meters).
        """
        depth_m = _depth_to_float_metres(depth)
        if depth_m.ndim == 3 and depth_m.shape[0] == 1:
            depth_m = depth_m[0]

        valid = depth_m > 0.0  # NaN / -inf / 0 -> False

        # Log transform (masked pixels -> 0 first so NaN cannot poison the
        # log/cast; +inf stays and clips to the max level below)
        depth_m = np.where(valid, depth_m, 0.0)
        z = np.log(depth_m + self.shift_m)

        # Normalize to [0.0, 1.0] and scale to 8-bit [0, 255]
        norm = np.clip((z - self.z_min) / (self.z_max - self.z_min), 0.0, 1.0)
        encoded = (norm * QUANT_MAX).astype(np.uint8)

        # Zero-out invalid/masked measurements
        return np.where(valid, encoded, 0)

    def decode(self, depth_uint8) -> np.ndarray:
        """uint8 [0, 255] -> (H, W) float32 depth in metres.

        Args:
            depth_uint8: (H, W) or (1, H, W) uint8 image in [0, 255].
        """
        arr = np.asarray(depth_uint8, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]

        # Normalize to [0.0, 1.0] and invert the log transform
        norm = arr.astype(np.float32) / QUANT_MAX
        depth_m = np.exp(norm * (self.z_max - self.z_min) + self.z_min) - self.shift_m

        # Zero-out invalid measurements (where input was zero)
        return np.where(arr > 0, depth_m, 0.0).astype(np.float32)