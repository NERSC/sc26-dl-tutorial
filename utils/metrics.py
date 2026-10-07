import torch

from distributed.mappings import allreduce_forward_identity_backward


@torch.jit.script
def _squared_error_sum(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.sum((pred - target) ** 2, dim=(-1, -2))


@torch.jit.script
def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    squared_error_sum = _squared_error_sum(pred, target)
    num_pixels = pred.shape[-2] * pred.shape[-1]
    return torch.mean(torch.sqrt(squared_error_sum / num_pixels), dim=0)


def rmse_spatial(pred, target, spatial_shard):
    squared_error_sum = _squared_error_sum(pred, target)
    squared_error_sum = allreduce_forward_identity_backward(
        squared_error_sum, "sp1-sp2"
    )
    num_pixels = (
        spatial_shard.global_shape[0] * spatial_shard.global_shape[1]
    )
    return torch.mean(torch.sqrt(squared_error_sum / num_pixels), dim=0)
