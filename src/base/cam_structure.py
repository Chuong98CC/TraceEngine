from dataclasses import dataclass
import json
import numpy as np


def read_calib_file(calib_file: str) -> dict:
    """Read camera calibration parameters from a JSON file.

    Supports three formats:
      1. Single-camera: flat dict with ``resolution``, ``intrinsics``, ``extrinsics``,
         ``distortions`` keys at the top level.
      2. Multi-camera: top-level keys are camera names (e.g. ``head_rgbd``,
         ``torso_rgbd``).  Used by ``astribot_camera_intrinsics.json``.
      3. Full calibration: has a ``camera`` key wrapping the multi-camera dict, plus
         optional ``lidar``.  Used by ``astribot_calibration_full.json``.

    Returns a dict with the following keys, each keyed by camera name:
        ``Intrinsics``   — ``{cam_name: 3×3 np.ndarray}``  (color intrinsics)
        ``Extrinsics``   — ``{cam_name: 4×4 np.ndarray}``
        ``Resolution``   — ``{cam_name: (w, h)}``
        ``Distortions``  — ``{cam_name: (5,) np.ndarray}``
        ``Baseline``     — ``{"camA->camB": float (mm), ...}``
                           Derived from extrinsics where ``parent_frame`` names another
                           known camera (i.e. a stereo pair).  The baseline is the L2
                           norm of the translation vector, converted from metres to mm.
    """
    with open(calib_file, "r") as f:
        data = json.load(f)

    cameras: dict = {}

    # ---- detect & normalise format ----
    if "camera" in data:
        # Full calibration file  (astribot_calibration_full.json)
        cameras = data["camera"]
    elif "intrinsics" in data and "extrinsics" in data:
        # Single-camera file  (e.g. astribot_calib_head_stereo_left.json)
        # Synthesise a camera name from the file stem.
        import os

        stem = os.path.splitext(os.path.basename(calib_file))[0]
        for prefix in ("astribot_calib_", "astribot_calibration_", "calib_"):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
        cameras = {stem: data}
    else:
        # Assume top-level keys are camera names  (astribot_camera_intrinsics.json)
        cameras = data

    # ---- parse every camera ----
    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    distortions: dict[str, np.ndarray] = {}
    resolution: dict[str, tuple[int, int]] = {}

    for cam_name, cam_data in cameras.items():
        # -- Intrinsics (prefer color, fall back to depth, then direct matrix) --
        intr_data = cam_data.get("intrinsics", {})
        if "color" in intr_data:
            mat = intr_data["color"]["matrix"]
            rows, cols = intr_data["color"]["rows"], intr_data["color"]["cols"]
        elif "depth" in intr_data:
            mat = intr_data["depth"]["matrix"]
            rows, cols = intr_data["depth"]["rows"], intr_data["depth"]["cols"]
        else:
            mat = intr_data["matrix"]
            rows, cols = intr_data["rows"], intr_data["cols"]
        intrinsics[cam_name] = np.array(mat).reshape(rows, cols)

        # -- Extrinsics (4×4) --
        extr_data = cam_data["extrinsics"]
        extr_matrix = np.array(extr_data["matrix"]).reshape(
            extr_data["rows"], extr_data["cols"]
        )
        extrinsics[cam_name] = extr_matrix

        # -- Distortions (5×1 is typical, but we keep it generic) --
        if "distortions" in cam_data:
            dist_data = cam_data["distortions"]
            distortions[cam_name] = np.array(dist_data["matrix"]).reshape(
                dist_data["rows"]
            )

        # -- Resolution --
        # Resolution can live at camera level or inside the intrinsics block.
        res_str = cam_data.get("resolution", intr_data.get("resolution", None))
        if res_str is not None:
            w_str, h_str = res_str.split("x")
            resolution[cam_name] = (int(w_str), int(h_str))
        else:
            resolution[cam_name] = (0, 0)

    # ---- derive stereo baselines from extrinsics ----
    baselines: dict[str, float] = {}
    for cam_name, extr in extrinsics.items():
        extr_data = cameras.get(cam_name, {}).get("extrinsics", {})
        parent_frame = extr_data.get("parent_frame", "")
        # If the parent frame is another known camera, we have a stereo pair.
        if parent_frame and parent_frame in intrinsics:
            pair = f"{parent_frame}->{cam_name}"
            t = extr[:3, 3]  # translation in metres
            baselines[pair] = float(np.linalg.norm(t) * 1000.0)  # m → mm

    return {
        "Intrinsics": intrinsics,
        "Extrinsics": extrinsics,
        "Resolution": resolution,
        "Baseline": baselines,
        "Distortions": distortions,
    }


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float   # focal length in pixels along x-axis
    fy: float   # focal length in pixels along y-axis
    cx: float   # principal point x-coordinate in pixels
    cy: float   # principal point y-coordinate in pixels
    bl: float   # baseline in meters, for stereo vision (distance between the two camera centers), None if monocular
    w:  int     # image width in pixels
    h:  int     # image height in pixels

    @staticmethod
    def from_intrinsics_matrix(K: np.ndarray, baseline: float = None, resolution: tuple[int, int] = None) -> "CameraIntrinsics":
        """Create CameraIntrinsics from a 3x3 intrinsics matrix K."""
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        if resolution is not None:
            w, h = resolution
        else:
            w, h = 0, 0
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, bl=baseline, w=w, h=h)

    @staticmethod
    def from_calib_file(calib_file: str, camera: str = "head_stereo_left", baseline_cams: str = "head_stereo_left->head_stereo_right") -> "CameraIntrinsics":
        cam = read_calib_file(calib_file)
        fx = cam["Intrinsics"][camera][0, 0]
        fy = cam["Intrinsics"][camera][1, 1]
        cx = cam["Intrinsics"][camera][0, 2]
        cy = cam["Intrinsics"][camera][1, 2]
        try:
            bl = cam["Baseline"][baseline_cams] / 1000.0  # convert from mm to meters
        except KeyError:
            bl = None
        w = cam["Resolution"][camera][0]
        h = cam["Resolution"][camera][1]
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, bl=bl, w=w, h=h)

    @property
    def fxy(self):
        return 0.5 * (self.fx + self.fy)

    def fx_scaled(self, resize_w: int):
        return self.fx * resize_w / self.w

    def fy_scaled(self, resize_h: int):
        return self.fy * resize_h / self.h

    def fxy_scaled(self, resize_w: int, resize_h: int):
        return 0.5 * (self.fx_scaled(resize_w) + self.fy_scaled(resize_h))

    def cx_scaled(self, resize_w: int):
        return self.cx * resize_w / self.w

    def cy_scaled(self, resize_h: int):
        return self.cy * resize_h / self.h

    def scaled_params(self, resize_w: int, resize_h: int):
        if resize_h and resize_w:
            fx = self.fx_scaled(resize_w)
            fy = self.fy_scaled(resize_h)
            cx = self.cx_scaled(resize_w)
            cy = self.cy_scaled(resize_h)
        else:
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        return fx,fy,cx,cy

    def intrinsic(self, resize_h: int=None, resize_w: int=None):
        """Return the 3x3 intrinsic matrix, optionally scaled to a different resolution."""
        fx, fy, cx, cy = self.scaled_params(resize_w, resize_h)
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def disparity_to_depth(self, disparity: np.ndarray, resize_w: int = None, resize_h: int = None, mask: np.ndarray = None) -> np.ndarray:
        """Convert disparity map to depth map using the formula: depth = (f * bl) / disparity."""
        with np.errstate(divide="ignore"):
            if resize_w is not None and resize_h is not None:
                fxy = self.fxy_scaled(resize_w, resize_h)
            else:
                fxy = self.fxy
            depth = self.bl * fxy / disparity

        if mask is not None:
            depth[~mask] = -1.0  # Apply mask
        else:
            depth[disparity <= 1e-2] = -1.0  # Set invalid disparities to -1

        return depth

    def back_project(self, uv: np.ndarray, depth: np.ndarray, resize_w: int=None, resize_h: int=None) -> np.ndarray:
        """Back-project 2D pixel coordinates (u, v) with depth to 3D points.

        ``uv`` and ``depth`` are expressed in a ``resize_w × resize_h`` image
        (e.g. the depth map / video frame), which may differ from the
        calibration resolution (``self.w × self.h``).  The intrinsics are
        scaled **per-axis** to that image before projecting, because the
        width/height ratios differ when the aspect ratio changes
        (e.g. 1280×720 → 640×480).
        """
        u = uv[:, 0]
        v = uv[:, 1]

        fx, fy, cx, cy = self.scaled_params(resize_w, resize_h)

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return np.stack((x, y, z), axis=-1)

    def forward_project(self, xyz: np.ndarray, resize_w: int=None, resize_h: int=None) -> np.ndarray:
        """Forward-project 3D points to 2D pixel coordinates (u, v).

        Coordinates are returned in a ``resize_w × resize_h`` image, using the
        same per-axis scaled intrinsics as :meth:`back_project` (its inverse).
        """
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = np.clip(xyz[:, 2], 1e-6, None)  # guard against divide-by-zero
        fx, fy, cx, cy = self.scaled_params(resize_w, resize_h)
        u = (x * fx) / z + cx
        v = (y * fy) / z + cy
        return np.stack((u, v), axis=-1)


@dataclass(slots=True)
class CameraExtrinsics:
    R: np.ndarray  # 3x3 rotation matrix
    T: np.ndarray  # 3x1 translation vector

    @staticmethod
    def transform_pointcloud(points: np.ndarray, extrinsics: "CameraExtrinsics") -> np.ndarray:
        """Apply the extrinsic transformation to the point cloud."""
        R, T = extrinsics.R, extrinsics.T
        return (R @ points.T).T + T.reshape(1, 3)

    @staticmethod
    def from_calib_file(calib_file: str, pair: str) -> "CameraExtrinsics":
        cam = read_calib_file(calib_file)
        extrinsic_matrix = cam["Extrinsics"][pair]
        R = extrinsic_matrix[:3, :3]
        T = extrinsic_matrix[:3, 3]
        return CameraExtrinsics(R=R, T=T)
