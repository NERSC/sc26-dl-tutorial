"""ConvNeXt with spatial halos and replicated weights (DP plus SP, no TP)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed.mappings import halo_forward_sum_backward
from networks.convnext import (
    ConvNeXtField, kernel_radius, model_kwargs,
)
from utils import comm


def exchange_width_stripes(x, halo):
    """Complete stripe example: attach columns from neighboring sp2 ranks."""
    return halo_forward_sum_backward(x, halo, -1, "sp2")


def exchange_spatial_halo(x, halo):
    """Exchange rows INCLUDING the received columns, so corners travel too."""
    x = exchange_width_stripes(x, halo)
    return halo_forward_sum_backward(x, halo, -2, "sp1")


def validate_spatial_shard(shard, img_size, patch_size, halo):
    """Check every tile, on every rank, before any halo communication."""
    if shard is None or shard.global_shape != tuple(img_size):
        raise ValueError("ConvNeXt needs a spatial shard matching img_size")
    for extent, splits, rank, group in zip(
        img_size, (shard.h_splits, shard.w_splits),
        (shard.h_rank, shard.w_rank), ("sp1", "sp2"),
    ):
        if (
            len(splits) != comm.get_size(group) or sum(splits) != extent
            or rank != comm.get_rank(group)
            or any(s <= 0 or s % patch_size for s in splits)
        ):
            raise ValueError(f"ConvNeXt {group} shards must match the mesh and be patch-aligned")
        if len(splits) > 1 and min(splits) // patch_size < halo:
            raise ValueError(f"ConvNeXt {group} needs at least {halo} feature cells per tile")


class HaloDepthwiseConv2d(nn.Conv2d):
    """Same parameter names as nn.Conv2d; only the input halo changes."""

    def __init__(self, channels, kernel_size=7):
        halo = kernel_radius(kernel_size)
        super().__init__(channels, channels, kernel_size, padding=halo, groups=channels)
        # Depthwise weights are (C, 1, K, K). Match the singleton-channel
        # stride produced by convolution backward before DDP builds its buckets.
        # This preserves element order and avoids a spurious DDP stride warning.
        nn.utils.convert_conv2d_weight_memory_format(self, torch.channels_last)
        self.halo = halo
        self.local_padding = tuple(
            halo if comm.get_size(group) == 1 else 0 for group in ("sp1", "sp2")
        )

    def forward(self, x):
        x = exchange_spatial_halo(x, self.halo)
        return F.conv2d(x, self.weight, self.bias, padding=self.local_padding, groups=self.groups)


class DistributedConvNeXt(ConvNeXtField):
    """ConvNeXtField whose depthwise convolutions exchange spatial halos."""

    def __init__(self, *, spatial_shard, **kwargs):
        if comm.get_size("tp") != 1:
            raise ValueError("ConvNeXt supports spatial parallelism only; use tp=1")
        super().__init__(**kwargs)
        validate_spatial_shard(spatial_shard, self.img_size, self.patch_size, self.halo)
        self.spatial_shard = spatial_shard
        self.input_shape = spatial_shard.local_shape
        for block in self.blocks:
            original = block.depthwise
            block.depthwise = HaloDepthwiseConv2d(self.embed_dim, self.kernel_size)
            block.depthwise.load_state_dict(original.state_dict())
        for parameter in self.parameters():
            parameter.comm_metadata = {
                "sharded": [None] * parameter.ndim,
                "shared": ["sp1-sp2"],
                "reduce": ["sp1-sp2"],
            }


def ConvNeXt(params, spatial_shard):
    """Build a spatially sharded ConvNeXt from a parameter object."""
    return DistributedConvNeXt(spatial_shard=spatial_shard, **model_kwargs(params))
