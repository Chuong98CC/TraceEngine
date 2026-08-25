import json
import numpy as np
from typing import Optional
from pathlib import Path
import cv2
import lz4.frame
# ---------------------------------------------------------------------------
# Paths relative to repo root
# ---------------------------------------------------------------------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "assets"
CALIB_PATH = DATA_ROOT / "astribot_cam_calib" / "astribot_calibration_full.json"
IMAGES_ROOT = DATA_ROOT / "astribot_test_imgs"
# ---------------------------------------------------------------------------
# Mapping: image directory suffix → calibration key
# ---------------------------------------------------------------------------
DIR_TO_CALIB: dict[str, str] = {
    "cam_head":                "head_rgbd",
    "cam_head_depth":          "head_depth",
    "cam_head_stereo_left":    "head_stereo_left",
    "cam_head_stereo_right":   "head_stereo_right",
    "cam_torso":               "torso_rgbd",
    "cam_left_wrist":          "left_wrist_rgbd",
    "cam_right_wrist":         "right_wrist_rgbd",
}
# Reverse: calibration key → image directory name
CALIB_TO_DIR: dict[str, str] = {v: k for k, v in DIR_TO_CALIB.items()}

CAMERA_SETS: dict[str, list[str]] = {
    "set0": ["head_stereo_left", "head_stereo_right"],
    "set1": ["head_stereo_left", "head_stereo_right", "head_rgbd"],
    "set2": ["head_stereo_left", "head_stereo_right", "head_rgbd", "torso_rgbd"],
}

CAMERA_DEPTH_CONFIG = {
    "head_rgbd": {
        "depth_shape": (960, 1280),
        "depth_scale": 0.001,
    },
    "torso_rgbd": {
        "depth_shape": (480, 640),
        "depth_scale": 0.001,
    },
    "left_wrist_rgbd": {
        "depth_shape": (360, 640),
        "depth_scale": 0.0001,
    },
    "right_wrist_rgbd": {
        "depth_shape": (360, 640),
        "depth_scale": 0.0001,
    },
}

CAMERAS = list(CAMERA_DEPTH_CONFIG.keys())


def _scale_intrinsics_matrix(
    matrix: list[float],
    orig_w: int,
    orig_h: int,
    target_w: int = 640,
    target_h: int = 480,
) -> np.ndarray:
    """Scale a 3x3 intrinsic matrix from original to target resolution."""
    K = np.array(matrix, dtype=np.float64).reshape(3, 3).copy()
    sx = target_w / orig_w
    sy = target_h / orig_h
    K[0, 0] *= sx  # fx
    K[0, 2] *= sx  # cx
    K[1, 1] *= sy  # fy
    K[1, 2] *= sy  # cy
    return K

def _parse_resolution(res) -> tuple[int, int]:
    """Parse a calibration resolution ('WxH' string or [w, h]) into (w, h)."""
    if isinstance(res, str):
        w, h = map(int, res.split("x"))
        return w, h
    return int(res[0]), int(res[1])

def _get_resolution(cam_data: dict) -> Optional[str]:
    """Extract resolution string from camera data (camera-level or intrinsics-level)."""
    if "resolution" in cam_data:
        return cam_data["resolution"]
    if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
        return cam_data["intrinsics"]["resolution"]
    return None

def load_calib(
    json_path: str | Path,
    target_res: tuple[int, int] = (640, 480),
) -> dict:
    """Load calibration JSON and scale all intrinsics to target resolution."""
    with open(json_path) as f:
        data = json.load(f)

    target_w, target_h = target_res
    target_res_str = f"{target_w}x{target_h}"

    cameras = data.get("camera", {})
    for cam_name, cam_data in cameras.items():
        res_str = _get_resolution(cam_data)
        if res_str is None:
            print(f"Warning: No resolution found for '{cam_name}', skipping.")
            continue

        orig_w, orig_h = map(int, res_str.split("x"))
        if orig_w == target_w and orig_h == target_h:
            cam_data["resolution"] = target_res_str
            if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
                cam_data["intrinsics"]["resolution"] = target_res_str
            continue

        if "intrinsics" in cam_data:
            for sensor_type, sensor in cam_data["intrinsics"].items():
                if sensor_type == "resolution":
                    continue
                if "matrix" not in sensor:
                    continue
                K_scaled = _scale_intrinsics_matrix(
                    sensor["matrix"], orig_w, orig_h, target_w, target_h
                )
                sensor["matrix"] = K_scaled.flatten().tolist()

        cam_data["resolution"] = target_res_str
        if "intrinsics" in cam_data and "resolution" in cam_data["intrinsics"]:
            cam_data["intrinsics"]["resolution"] = target_res_str

    return data

def get_camera_params(
    calib: dict,
    camera_names: list[str],
    sensor_type: str = "color",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract extrinsics (N,4,4) and intrinsics (N,3,3) for given cameras.

    Args:
        calib: Calibration dictionary.
        camera_names: Camera names to fetch in order.
        sensor_type: Intrinsic sensor type key.
        to_world_to_camera: If True, convert loaded camera-to-world extrinsics
            into world-to-camera before returning.
    """
    exts, ixts, resolutions = [], [], []
    for name in camera_names:
        cam = calib["camera"][name]
        exts.append(np.array(cam["extrinsics"]["matrix"], dtype=np.float64).reshape(4, 4))
        ixts.append(np.array(cam["intrinsics"][sensor_type]["matrix"], dtype=np.float64).reshape(3, 3))
        # The resolution the ext/ixt are recorded at lives at the intrinsics
        # level ('WxH' string, rewritten to the target by load_calib), with
        # optional per-sensor or camera-level overrides.
        sensor = cam["intrinsics"].get(sensor_type, {})
        res = sensor.get("resolution") or cam["intrinsics"].get("resolution") or cam.get("resolution")
        if res is None:
            raise ValueError(f"No resolution recorded for camera '{name}'")
        resolutions.append(np.array(_parse_resolution(res), dtype=np.int32))
    return np.stack(exts), np.stack(ixts), resolutions

def load_depth_lz4(depth_path: Path, shape: tuple[int, int]) -> np.ndarray:
    raw = depth_path.read_bytes()
    try:
        decoded = lz4.frame.decompress(raw)
    except lz4.frame.LZ4FrameError as exc:
        raise RuntimeError(f"Failed to decompress depth file {depth_path}: {exc}") from exc
    arr = np.frombuffer(decoded, dtype=np.uint16)
    expected = shape[0] * shape[1]
    if arr.size == expected:
        return arr.reshape(shape)
    if arr.size == expected * 3:
        arr = arr.reshape((shape[0], shape[1], 3)).astype(np.float32).mean(axis=2)
        return arr.astype(np.uint16)
    raise ValueError(f"Unexpected decoded depth size for {depth_path}: got {arr.size}, expected {expected} or {expected*3}")

def colorize_depth(depth_frame: np.ndarray, max_depth_m: float = 5.0) -> np.ndarray:
    """
    Convert a depth frame to a grayscale image using log-compressed, fixed normalization.

    Mapping:
      - Zero / invalid pixels (NaN, Inf, 0)  → black (0)
      - Closest valid depth                  → dark grey (30)
      - Furthest valid depth (max_depth_m)   → light grey (240)

    Log compression (log1p) is applied so that near-field detail is preserved and
    the same depth value always maps to the same grey level across all frames
    (fixed scale = log1p(max_depth_m), not per-frame min/max).

    Args:
        depth_frame: (H, W) array in metres (float32) or millimetres (uint16 / large float).
                     Unit auto-detected: if median of valid pixels > 100, assumed mm.
        max_depth_m: far-plane clip distance in metres.

    Returns:
        (H, W, 3) uint8 BGR image (all channels equal → greyscale).
    """
    depth_f = depth_frame.astype(np.float32)

    # Replace NaN/Inf with 0 (treated as invalid below)
    depth_f = np.where(np.isfinite(depth_f), depth_f, 0.0)

    # Auto-detect mm vs metres: median of non-zero pixels > 100 → mm
    valid = depth_f[depth_f > 0.0]
    if valid.size > 0 and float(np.median(valid)) > 100.0:
        depth_f /= 1000.0

    valid_mask = depth_f > 0.0

    # Log-compress then normalize with fixed scale (consistent across frames)
    clipped = np.clip(depth_f, 0.0, max_depth_m)
    normalized = np.log1p(clipped) / np.log1p(max_depth_m)  # → [0, 1]

    # dark grey (30) = close, light grey (240) = far; black (0) = invalid
    grey = np.where(
        valid_mask,
        (30 + normalized * 210).astype(np.uint8),
        np.uint8(0),
    )

    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

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


def load_rgbd(frame_idx: int, camera_name: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load RGB and depth images for a given frame index.

    Returns:
        (rgb_path, depth_metrics, extrinsics, intrinsics, resolution)
        where ``resolution`` is the (w, h) the intrinsics/extrinsics are
        recorded at — the intrinsics are scaled to depth resolution 0 by ``load_calib``,
        so callers resizing the RGB image should scale the intrinsics from
        this resolution to the target via ``_scale_intrinsics_matrix``.
    """
    assert camera_name in CAMERA_DEPTH_CONFIG, f"Unknown camera: {camera_name}"
    rgb_path = f"{IMAGES_ROOT}/{camera_name}/color/img_{frame_idx:06d}.jpg"
    depth_path = f"{IMAGES_ROOT}/{camera_name}/depth/img_{frame_idx:06d}.jpg"
    depth_shape = CAMERA_DEPTH_CONFIG[camera_name]["depth_shape"]
    depth_scale = CAMERA_DEPTH_CONFIG[camera_name]["depth_scale"]

    depth_raw = load_depth_lz4(Path(depth_path), shape=depth_shape)
    depth_metrics = depth_raw.astype(np.float32) * depth_scale
    depth_metrics[depth_metrics <= 0.0] = 0.0
    depth_metrics[depth_metrics > 1.0] = 0.0

    calib = load_calib(CALIB_PATH)
    exts, ixts, resolutions = get_camera_params(calib, [camera_name], sensor_type="color")
    return rgb_path, depth_metrics, exts[0], ixts[0], resolutions[0]