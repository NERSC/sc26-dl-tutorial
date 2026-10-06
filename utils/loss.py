import torch

from distributed.mappings import allreduce_forward_identity_backward


@torch.jit.script
def _squared_error_sum(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.sum((pred - target) ** 2)


def mse_loss(
    pred: torch.Tensor, target: torch.Tensor, global_shape
) -> torch.Tensor:
    squared_error = _squared_error_sum(pred, target)
    squared_error = allreduce_forward_identity_backward(
        squared_error, "sp1-sp2"
    )
    global_elements = (
        pred.shape[0]
        * pred.shape[1]
        * global_shape[0]
        * global_shape[1]
    )
    return squared_error / global_elements
