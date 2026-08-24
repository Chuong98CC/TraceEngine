
import torch
from dataclasses import fields, is_dataclass
from functools import wraps

def convert_tensor(x, allow_cast: bool):
    if not isinstance(x, torch.Tensor):
        return x
    if x.dtype not in [torch.float16, torch.bfloat16]:
        return x
    if allow_cast:
        return x.to(dtype=torch.float32)
    else:
        raise TypeError(f"Tensor of dtype {x.dtype} is not allowed. Expected float32.")
    return x

def convert_tensor_struct(data, allow_cast: bool):
    if isinstance(data, torch.Tensor):
        return convert_tensor(data, allow_cast)
    elif isinstance(data, list):
        return [convert_tensor_struct(item, allow_cast) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_tensor_struct(item, allow_cast) for item in data)
    elif isinstance(data, dict):
        return {key: convert_tensor_struct(value, allow_cast) for key, value in data.items()}
    elif is_dataclass(data):
        return type(data)(**{field.name: convert_tensor_struct(getattr(data, field.name), allow_cast) for field in fields(data)})
    else:
        return data

def ensure_float32(allow_cast: bool = True):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_args = tuple(convert_tensor_struct(arg, allow_cast) for arg in args)
            new_kwargs = {key: convert_tensor_struct(value, allow_cast) for key, value in kwargs.items()}
            with torch.autocast(device_type='cuda', enabled=False):
                return convert_tensor_struct(func(*new_args, **new_kwargs), True)
        return wrapper
    return decorator

@ensure_float32(allow_cast=False)
def batch_unproject(depth: torch.Tensor, intrinsic: torch.Tensor, extrinsic: torch.Tensor, channel_last: bool = False) -> torch.Tensor:
    if depth.ndim == 4:
        pcds = batch_unproject(
            depth.view(-1, *depth.shape[2:]),
            intrinsic.view(-1, *intrinsic.shape[2:]),
            extrinsic.view(-1, *extrinsic.shape[2:]),
            channel_last
        )
        return pcds.reshape(depth.shape[:2] + pcds.shape[1:])
    assert depth.ndim == 3, "depth must be a 3D array"
    t, h, w = depth.shape
    assert intrinsic.shape == (t, 3, 3), f"intrinsic must be a 3D array of shape (t, 3, 3), got {intrinsic.shape}"
    assert extrinsic.shape == (t, 4, 4), f"extrinsic must be a 3D array of shape (t, 4, 4), got {extrinsic.shape}"

    # Generate 3D point cloud in world coordinates
    v, u = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
    uv_homogeneous = torch.stack((u, v, torch.ones_like(u)), dim=-1).float()
    K_inv = torch.linalg.inv(intrinsic)
    camera_coords = torch.einsum("nij, xyj -> nxyi", K_inv, uv_homogeneous)
    camera_coords = camera_coords * depth[..., None]
    camera_coords = torch.cat((camera_coords, torch.ones_like(camera_coords[..., :1])), dim=-1)

    inv_extrinsics = torch.linalg.inv(extrinsic)
    world_coordinates = torch.einsum("nij, nxyj -> nxyi", inv_extrinsics, camera_coords)

    if channel_last:
        return world_coordinates[..., :3]
    else:
        return world_coordinates[..., :3].permute(0, 3, 1, 2)

@ensure_float32(allow_cast=False)
def batch_project(
    pts3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    pts3d_homogeneous = torch.cat((pts3d, torch.ones_like(pts3d[..., :1])), dim=-1) # (n, 4)

    pts_camera_homogeneous = torch.einsum("...ij, ...j -> ...i", extrinsics, pts3d_homogeneous) # (n, 4)
    pts_camera = pts_camera_homogeneous[..., :3] / pts_camera_homogeneous[..., 3:] # (n, 3)

    pts_image_homogeneous = torch.einsum("...ij, ...j -> ...i", intrinsics, pts_camera) # (n, 3)
    pts_image = pts_image_homogeneous[..., :2] / pts_image_homogeneous[..., 2:] # (n, 2)

    return pts_image