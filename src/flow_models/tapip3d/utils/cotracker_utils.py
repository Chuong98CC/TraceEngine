import torch
import torch.nn.functional as F
from typing import Optional, Tuple

# https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/cotracker/cotracker3_online.py

def posenc(x, *, min_deg=None, max_deg=None, scales=None):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
      encoded: torch.Tensor, encoded variables.
    """
    if scales is None and min_deg is not None and min_deg == max_deg:
        return x
    if scales is None:
        assert min_deg is not None and max_deg is not None
        scales = torch.tensor(
            [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
        )
    else:
        assert min_deg is None and max_deg is None
    # scales = 2 ** torch.arange(min_deg, max_deg, device=x.device, dtype=x.dtype)

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)

def bilinear_sampler(input, coords, mode="bilinear", align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = torch.stack([coords[..., 1], coords[..., 2], coords[..., 0]], dim=-1)

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode, mode=mode
    )

def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    if embed_dim % 2 != 0:
        result = get_1d_sincos_pos_embed_from_grid(embed_dim - 1, pos)
        return torch.cat([result, result.new_zeros(result.shape[:-1] + (1,))], dim=-1)

    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

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