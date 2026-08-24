import sys
import numpy as np
from typing import Optional
import cv2

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