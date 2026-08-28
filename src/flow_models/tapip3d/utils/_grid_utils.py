# Vendored from <repo>/utils/inference_utils.py @ 9359ae2 (get_grid_queries,
# lines 12-43). Verbatim; import rewiring only (get_points_on_a_grid now comes
# from ._sampling; the source module's other helpers/imports are not vendored).
# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
import torch

from ._sampling import get_points_on_a_grid

def get_grid_queries(grid_size: int, depths: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor):
    if len (depths.shape) == 3:
        return get_grid_queries(
            grid_size=grid_size,
            depths=depths.unsqueeze(0),
            intrinsics=intrinsics.unsqueeze(0),
            extrinsics=extrinsics.unsqueeze(0)
        ).squeeze(0)

    image_size = depths.shape[-2:]
    xy = get_points_on_a_grid(grid_size, image_size).to(intrinsics.device) # type: ignore
    ji = torch.round(xy).to(torch.int32)
    d = depths[:, 0][torch.arange(depths.shape[0])[:, None], ji[..., 1], ji[..., 0]]

    assert d.shape[0] == 1, "batch size must be 1"
    mask = d[0] > 0
    d = d[:, mask]
    xy = xy[:, mask]
    ji = ji[:, mask]

    inv_intrinsics0 = torch.linalg.inv(intrinsics[0, 0])
    inv_extrinsics0 = torch.linalg.inv(extrinsics[0, 0])

    xy_homo = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    xy_homo = torch.einsum('ij,bnj->bni', inv_intrinsics0, xy_homo)
    local_coords = xy_homo * d[..., None]
    local_coords_homo = torch.cat([local_coords, torch.ones_like(local_coords[..., :1])], dim=-1)
    world_coords = torch.einsum('ij,bnj->bni', inv_extrinsics0, local_coords_homo)
    world_coords = world_coords[..., :3]

    queries = torch.cat([torch.zeros_like(xy[:, :, :1]), world_coords], dim=-1).to(depths.device)  # type: ignore
    return queries
