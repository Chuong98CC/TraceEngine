# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)
"""Correlation prep phases and pointops2 KNN for the ONNX wrapper.

Simplified copy of models/corr_features/knn_feature_4d_optimized.py: only the
surface Tapip3D_ONNX / StreamONNX use (pyramid building, support-feature
preparation, KNN). The trained per-iteration forward (NeighborTransformer,
posenc MLPs, LayerNorm) runs inside the corr_forward ONNX graph and does not
exist here, so no weights are ever needed. Kept methods are verbatim copies
of the original except: the better_depth_downsample forks collapse to the
nearest-exact path, and the use_local_pos_input branch (which can never
execute on the ONNX path) is deleted.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from einops import rearrange, repeat
import torch.nn as nn

from .common_utils import ensure_float32
from .cotracker_utils import bilinear_sampler

import third_party.pointops2.functions.pointops as pointops


def _get_index_offset_for_knnquery(batch_indices, dtype=torch.long):
    """A helper function to get index offset from ordered
    batch indices.

    Args:
        batch_indices: a tensor of shape (B,), where each element
            denotes the batch index.
    Returns:
        a tensor of shape (max_batch_index + 1,) where each element
            denotes the cumsum of number of elements belonging to the batch
    """
    assert (batch_indices[1:] >= batch_indices[:-1]).all()
    max_batch_index = batch_indices.max()

    offset = torch.zeros(
        (max_batch_index + 1, ), dtype=dtype, device=batch_indices.device
    )
    inds, counts = torch.unique_consecutive(batch_indices, return_counts=True)
    offset[inds] = counts
    existence = offset > 0

    offset = torch.cumsum(offset, dim=-1)

    return offset, existence

@dataclass
class KNNQueryInput:
    context_coords: torch.Tensor
    query_coords: torch.Tensor
    context_batch_offsets: torch.Tensor
    query_batch_offsets: torch.Tensor

    def __post_init__(self):
        assert self.context_coords.dtype == torch.float32
        assert self.query_coords.dtype == torch.float32
        assert self.context_batch_offsets.dtype == torch.int32
        assert self.query_batch_offsets.dtype == torch.int32
        assert self.context_batch_offsets.shape[0] == self.query_batch_offsets.shape[0]
        assert len (self.context_coords.shape) == 2 and len (self.query_coords.shape) == 2
        assert len (self.context_batch_offsets.shape) == 1 and len (self.query_batch_offsets.shape) == 1

    def contiguous(self) -> 'KNNQueryInput':
        return KNNQueryInput(
            context_coords=self.context_coords.contiguous(),
            query_coords=self.query_coords.contiguous(),
            context_batch_offsets=self.context_batch_offsets.contiguous(),
            query_batch_offsets=self.query_batch_offsets.contiguous(),
        )

def smart_tensor_op(fn: Callable[[torch.Tensor], torch.Tensor], x: Any) -> Any:
    if torch.is_tensor(x):
        return fn(x)
    elif isinstance(x, list):
        return [smart_tensor_op(fn, item) for item in x]
    elif isinstance(x, tuple):
        return tuple(smart_tensor_op(fn, item) for item in x)
    elif isinstance(x, dict):
        return {k: smart_tensor_op(fn, v) for k, v in x.items()}
    else:
        return x

@dataclass
class CorrContext:
    projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    feat_pyramid: Optional[List[torch.Tensor]] = None
    pcd_pyramid: Optional[List[torch.Tensor]] = None
    depth_pyramid: Optional[List[torch.Tensor]] = None
    query_feat_pyramid: Optional[List[torch.Tensor]] = None
    query_support_offset_pyramid: Optional[List[torch.Tensor]] = None
    query_support_feat_pyramid: Optional[List[torch.Tensor]] = None
    query_support_ti_pyramid: Optional[List[torch.Tensor]] = None

    def verify_window(self) -> bool:
        for field in fields(self):
            if getattr(self, field.name) is None:
                return False
        return True

    def verify_shared(self) -> bool:
        return self.query_support_ti_pyramid is not None

    def time_slice(self, start: int, end: int) -> 'CorrContext':
        assert self.verify_window()
        return CorrContext(
            projector=lambda x: self.projector(torch.cat([x[..., :1] + start, x[..., 1:]], dim=-1)), # type: ignore
            feat_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.feat_pyramid),
            pcd_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.pcd_pyramid),
            depth_pyramid=smart_tensor_op(lambda x: x[:, start:end], self.depth_pyramid),
            query_feat_pyramid=self.query_feat_pyramid,
            query_support_offset_pyramid=self.query_support_offset_pyramid,
            query_support_feat_pyramid=self.query_support_feat_pyramid,
        )

    def select_queries(self, mask: Union[torch.Tensor, List[int], slice]) -> 'CorrContext':
        self_dict = self.__dict__.copy()
        if isinstance(mask, torch.Tensor):
            assert len(mask.shape) == 1
        if self.query_feat_pyramid is not None:
            self_dict["query_feat_pyramid"] = [x[:, mask] for x in self.query_feat_pyramid] # type: ignore
        if self.query_support_offset_pyramid is not None:
            self_dict["query_support_offset_pyramid"] = [x[:, mask] for x in self.query_support_offset_pyramid] # type: ignore
        if self.query_support_feat_pyramid is not None:
            self_dict["query_support_feat_pyramid"] = [x[:, mask] for x in self.query_support_feat_pyramid] # type: ignore
        if self.query_support_ti_pyramid is not None:
            self_dict["query_support_ti_pyramid"] = [x[:, mask] for x in self.query_support_ti_pyramid] # type: ignore
        return CorrContext(**self_dict)

    def copy(self) -> 'CorrContext':
        return CorrContext(**{k: v for k, v in self.__dict__.items() if v is not None})

class CorrPrepProcessor(nn.Module):
    """Prep-only subset of KNNCorrFeature4D_Optimized for the ONNX wrapper.

    Builds the per-window correlation context (pcd/feat pyramids, query and
    support features) and runs the pointops2 KNN. It has no trainable
    parameters — the trained per-iteration math runs inside the corr_forward
    ONNX graph. Downsampling is nearest-exact (the original's bilinear path
    was deprecated); use_local_pos_input is not supported.
    """
    def __init__(
        self,
        image_size: Tuple[int, int],
        corr_levels: int,
        k_neighbors: int,
    ):
        super().__init__()

        self.image_size = image_size
        self.corr_levels = corr_levels
        self.k_neighbors = k_neighbors

    @torch.no_grad()
    def multi_knnquery(self, k_neighbors: int, inputs: List[KNNQueryInput]) -> List[torch.Tensor]:
        context_offsets, query_offsets = [], []
        cum_context_offset, cum_query_offset = 0, 0

        for input in inputs:
            context_offsets.append(cum_context_offset)
            query_offsets.append(cum_query_offset)
            cum_context_offset += input.context_coords.shape[0]
            cum_query_offset += input.query_coords.shape[0]
        query_input = KNNQueryInput(
            context_coords=torch.cat([input.context_coords for input in inputs], dim=0),
            query_coords=torch.cat([input.query_coords for input in inputs], dim=0),
            context_batch_offsets=torch.cat([input.context_batch_offsets + context_offsets[i] for i, input in enumerate(inputs)], dim=0),
            query_batch_offsets=torch.cat([input.query_batch_offsets + query_offsets[i] for i, input in enumerate(inputs)], dim=0),
        )
        query_input = query_input.contiguous()
        knn_idx, knn_dist = pointops.knnquery(k_neighbors, query_input.context_coords, query_input.query_coords, query_input.context_batch_offsets, query_input.query_batch_offsets) # type: ignore

        knn_idxs = []
        for i in range(len(inputs)):
            knn_idxs.append(knn_idx[query_offsets[i]:query_offsets[i] + inputs[i].query_coords.shape[0]] - context_offsets[i])

        return knn_idxs

    # we may want to figure out a better way to downsample the pointcloud later
    def _build_pyramid(self, pcds: torch.Tensor, feats: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        B, T = pcds.shape[:2]

        # Downsample the pointcloud to the resolution of feature map
        # (nearest-exact; bilinear downsampling was deprecated in the original)
        pcds = rearrange(pcds, "b t c h w -> (b t) c h w")
        pcds_raw = pcds.clone()
        pcds = torch.nn.functional.interpolate(pcds, size=feats[0].shape[-2:], mode='nearest-exact')
        pcds = rearrange(pcds, "(b t) c h w -> b t c h w", b=B, t=T)

        pcds_pyramid, feats_pyramid = [pcds], [feats]

        for level in range(self.corr_levels - 1):
            last_pcd = rearrange(pcds_pyramid[-1], 'b t c h w -> (b t) c h w')
            pcds_pyramid.append(rearrange(torch.nn.functional.interpolate(pcds_raw, size=(last_pcd.shape[-2] // 2, last_pcd.shape[-1] // 2), mode='nearest-exact'), '(b t) c h w -> b t c h w', b=B, t=T))
            last_feat = rearrange(feats_pyramid[-1], 'b t c h w -> (b t) c h w')
            feats_pyramid.append(rearrange(torch.nn.functional.avg_pool2d(last_feat, kernel_size=2, stride=2), '(b t) c h w -> b t c h w', b=B, t=T)) # type: ignore

        return pcds_pyramid, feats_pyramid # type: ignore

    @torch.no_grad()
    def prepare_shared_support_ti_singlepass(self, ctx: CorrContext, pcd_pyramid: List[torch.Tensor], queries: torch.Tensor) -> CorrContext:
        B, T = pcd_pyramid[0].shape[:2]
        N = queries.shape[1]
        query_frames = queries[..., 0].long()
        query_indices = torch.argsort(query_frames, dim=-1)

        queries_sorted = queries[torch.arange(B)[:, None], query_indices]
        query_batch_indices = T * torch.arange(B, device=pcd_pyramid[0].device, dtype=torch.long)[:, None] + queries_sorted[..., 0].long()
        query_batch_offsets, existence = _get_index_offset_for_knnquery(query_batch_indices)
        num_exist = existence.long().sum().item()

        # filter out empty batches
        query_batch_offsets = torch.unique_consecutive(query_batch_offsets).to(torch.int32)
        if query_batch_offsets[0] == 0:
            query_batch_offsets = query_batch_offsets[1:]

        query_knn_coords = queries_sorted[..., 1:].reshape(-1, 3)
        knn_queries: List[KNNQueryInput] = []

        ctx.query_support_ti_pyramid = []
        for pcds_level in pcd_pyramid:

            context_batch_offsets = torch.arange(num_exist, device=pcd_pyramid[0].device, dtype=torch.int32) + 1
            hw = pcds_level.shape[-1] * pcds_level.shape[-2]
            context_batch_offsets = (context_batch_offsets * hw).reshape(-1)
            context_coords = rearrange(pcds_level, 'b t c h w -> (b t) (h w) c')[: existence.shape[0]][existence]
            context_coords = context_coords.reshape(-1, 3)
            assert context_coords.dtype == torch.float32
            assert query_knn_coords.dtype == torch.float32

            knn_queries.append(KNNQueryInput(
                context_coords=context_coords,
                query_coords=query_knn_coords,
                context_batch_offsets=context_batch_offsets,
                query_batch_offsets=query_batch_offsets,
            ))

        knn_idxs = self.multi_knnquery(self.k_neighbors, knn_queries)

        for i in range(len(knn_idxs)):
            knn_idx = knn_idxs[i]
            pcds_level = pcd_pyramid[i]
            hw = pcds_level.shape[-1] * pcds_level.shape[-2]
            knn_idx = knn_idx % hw
            assert hw >= self.k_neighbors
            support_ti_ = repeat(query_frames, "b n -> b n k 2", k=self.k_neighbors).to(torch.float32).contiguous()
            support_ti_[torch.arange(B)[:, None], query_indices, :, 1] = rearrange(knn_idx, "(b n) k -> b n k", b=B).to(support_ti_.dtype)
            ctx.query_support_ti_pyramid.append(support_ti_)
        return ctx

    def prepare_shared(self, pcds: torch.Tensor, feats: torch.Tensor, queries: torch.Tensor) -> CorrContext:
        ctx = CorrContext()
        pcd_pyramid, feat_pyramid = self._build_pyramid(pcds, feats) # type: ignore
        ctx = self.prepare_shared_support_ti_singlepass(ctx, pcd_pyramid, queries)
        assert ctx.verify_shared()
        return ctx

    @ensure_float32(allow_cast=False)
    def prepare_window(self, *, shared_ctx: CorrContext, feats: torch.Tensor, pcds: torch.Tensor, queries: torch.Tensor, camera_locs: torch.Tensor, projector: Callable[[torch.Tensor], torch.Tensor]) -> CorrContext:
        ctx = shared_ctx.copy()
        assert ctx.verify_shared()

        ctx.projector = projector

        B, T = feats.shape[:2]
        N = queries.shape[1]

        ctx.pcd_pyramid, ctx.feat_pyramid = self._build_pyramid(pcds, feats) # type: ignore
        assert ctx.feat_pyramid is not None and ctx.pcd_pyramid is not None and ctx.query_support_ti_pyramid is not None

        ctx.depth_pyramid = []
        ctx.query_feat_pyramid = []
        ctx.query_support_offset_pyramid = []
        ctx.query_support_feat_pyramid = []
        query_coords = queries[..., 1:] # (B, N, 3)

        query_coords_2d_raw = projector(queries)[..., :2]
        for i in range(len(ctx.feat_pyramid)):
            query_coords_2d = query_coords_2d_raw.clone()
            query_coords_2d[..., 0] *= ctx.feat_pyramid[i].shape[-1] / self.image_size[-1]
            query_coords_2d[..., 1] *= ctx.feat_pyramid[i].shape[-2] / self.image_size[-2]
            query_txy = torch.cat([
                queries[..., 0:1],
                query_coords_2d,
            ], dim=-1)

            pcds = ctx.pcd_pyramid[i] # b t c h w
            feats = ctx.feat_pyramid[i] # b t c h w

            h, w = pcds.shape[-2:]

            interpolated_feats = rearrange(bilinear_sampler(
                rearrange(feats, 'b t c h w -> b c t h w'),
                rearrange(query_txy, 'b n c -> b n 1 1 c')
            ), 'b c n 1 1 -> b n c')

            projected_pcds = rearrange(ctx.projector(
                torch.cat(
                    [
                        repeat(torch.arange(T, device=pcds.device, dtype=pcds.dtype), 't -> b (t h w) 1', b=B, h=h, w=w),
                        rearrange(pcds, 'b t c h w -> b (t h w) c', b=B),
                    ],
                    dim=-1
                )
            ), 'b (t h w) c -> b t h w c', b=B, t=T, h=h, w=w)
            pcds_depth = projected_pcds[..., -1:]
            ctx.depth_pyramid.append(pcds_depth)

            # bilinear_sampler treats the coordinates as (x, y).
            # So we need to flip the coordinates to (i, t)
            support_features = rearrange(bilinear_sampler(
                rearrange(feats, 'b t c h w -> b c t (h w)'),
                rearrange(ctx.query_support_ti_pyramid[i].flip(-1), 'b n k c -> b (n k) 1 c'),
                mode="nearest"
            ), 'b c (n k) 1 -> b n k c', n=N)
            support_coords = rearrange(bilinear_sampler(
                rearrange(pcds, 'b t c h w -> b c t (h w)'),
                rearrange(ctx.query_support_ti_pyramid[i].flip(-1), 'b n k c -> b (n k) 1 c'),
                mode="nearest"
            ), 'b c (n k) 1 -> b n k c', n=N)

            # add the query point itself as the first support point
            support_coords = torch.cat([
                rearrange(query_coords, 'b n c -> b n 1 c'),
                support_coords,
            ], dim=2)
            support_features = torch.cat([
                rearrange(interpolated_feats, 'b n c -> b n 1 c'),
                support_features,
            ], dim=2)

            support_offsets = support_coords - support_coords[:, :, :1]

            ctx.query_support_feat_pyramid.append(support_features)
            ctx.query_support_offset_pyramid.append(support_offsets)
            ctx.query_feat_pyramid.append(interpolated_feats)

        assert ctx.verify_window()
        return ctx
