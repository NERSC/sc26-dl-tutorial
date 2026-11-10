"""Root-mean-square error over a field or a spatially sharded field."""

import torch

from distributed.mappings import allreduce_forward_identity_backward


def _squared_error_sum(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.sum((pred - target) ** 2, dim=(-1, -2))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the channel-wise RMSE of a complete field."""
    squared_error_sum = _squared_error_sum(pred, target)
    num_pixels = pred.shape[-2] * pred.shape[-1]
    return torch.mean(torch.sqrt(squared_error_sum / num_pixels), dim=0)


def rmse_spatial(pred, target, spatial_shard):
    """Return the channel-wise RMSE after summing squared error across spatial ranks."""
    squared_error_sum = _squared_error_sum(pred, target)
    squared_error_sum = allreduce_forward_identity_backward(
        squared_error_sum, "sp1-sp2"
    )
    num_pixels = (
        spatial_shard.global_shape[0] * spatial_shard.global_shape[1]
    )
    return torch.mean(torch.sqrt(squared_error_sum / num_pixels), dim=0)
