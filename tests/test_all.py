"""Master distributed-equivalence test.

Run with ``TEST_MODEL=vit``, ``TEST_MODEL=swin``, or ``TEST_MODEL=fno``.
"""

import os
import unittest

import torch
import torch.distributed as dist
from parameterized import parameterized

from distributed.helpers import init_params_for_shared_weights
from distributed.mappings import (
    allgather_forward_split_backward,
    init_ddp_model_and_reduction_hooks,
    split_forward_allgather_backward,
)
from distributed.spatial_sharding import (
    make_grid_spatial_shard,
    make_swin_spatial_shard,
    make_vit_spatial_shard,
)
from networks.fno import FourierNeuralOperator
from networks.fno_mp import FourierNeuralOperator as DistributedFNO
from networks.swin import SwinTransformer
from networks.swin_mp import DistributedSwinTransformer
from networks.vit import VisionTransformer
from networks.vit_mp import VisionTransformer as DistributedVisionTransformer
from tests.helpers import (
    DistributedTestCase,
    assert_reference_gradients,
    copy_reference_parameters,
)
from utils import comm
from utils.loss import mse_loss


MODEL = os.getenv("TEST_MODEL", "vit").lower()
if MODEL not in {"vit", "swin", "fno"}:
    raise ValueError("TEST_MODEL must be one of: vit, swin, fno")

CASES = {
    "vit": [
        # (name, batch/DP rank, H, W, patch, channels, embed, heads, depth, tolerance)
        ("small", 4, 128, 128, 8, 4, 512, 8, 2, 1e-5),
        ("rectangular", 4, 360, 720, 8, 20, 512, 8, 2, 1e-5),
    ],
    "swin": [
        # (name, batch/DP rank, H, W, patch, channels, embed, heads, depth, tolerance)
        # Swin's window size is fixed to (4, 4) in build_models().
        ("small", 4, 128, 128, 4, 4, 64, 8, 2, 1e-5),
        ("rectangular", 4, 256, 512, 4, 4, 64, 8, 2, 1e-5),
    ],
    "fno": [
        # (name, batch/DP rank, H, W, unused, channels, embed, unused, depth, tolerance)
        # FNO has no patch size or attention heads; those tuple positions are placeholders.
        ("even", 4, 64, 64, 1, 4, 32, 1, 2, 1e-5),
    ],
}


def build_models(model_name, height, width, patch_size, channels, embed, heads, depth):
    """Build matching serial/distributed models and this rank's spatial shard."""
    mesh = (
        comm.get_size("sp1"),
        comm.get_rank("sp1"),
        comm.get_size("sp2"),
        comm.get_rank("sp2"),
    )
    if model_name == "vit":
        kwargs = dict(
            img_size=(height, width),
            patch_size=patch_size,
            in_chans=channels,
            out_chans=channels,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            qkv_bias=True,
        )
        shard = make_vit_spatial_shard(
            (height, width), patch_size, *mesh, "tp-sp1-sp2-dp"
        )
        return (
            VisionTransformer(**kwargs),
            DistributedVisionTransformer(**kwargs, spatial_shard=shard),
            shard,
        )
    if model_name == "swin":
        window_size = (4, 4)
        kwargs = dict(
            img_size=(height, width),
            patch_size=patch_size,
            window_size=window_size,
            in_chans=channels,
            out_chans=channels,
            embed_dim=embed,
            depth=depth,
            num_heads=heads,
            qkv_bias=True,
        )
        shard = make_swin_spatial_shard(
            (height, width), patch_size, window_size, *mesh, "tp-sp1-sp2-dp"
        )
        return (
            SwinTransformer(**kwargs),
            DistributedSwinTransformer(**kwargs, spatial_shard=shard),
            shard,
        )

    kwargs = dict(
        img_size=(height, width),
        in_chans=channels,
        out_chans=channels,
        embed_dim=embed,
        depth=depth,
        modes_kh=min(12, height),
        modes_kw=min(17, width // 2 + 1),
    )
    shard = make_grid_spatial_shard((height, width), *mesh)
    return FourierNeuralOperator(**kwargs), DistributedFNO(**kwargs), shard


def gather_field(tensor, shard):
    tensor = allgather_forward_split_backward(
        tensor, -1, shard.w_splits, "sp2"
    )
    tensor = allgather_forward_split_backward(
        tensor, -2, shard.h_splits, "sp1"
    )
    return allgather_forward_split_backward(tensor, 0, None, "dp")


def error_metrics(actual, expected):
    """Return the worst relative L2 and absolute errors across all ranks."""
    difference = actual.detach() - expected.detach()
    denominator = expected.detach().norm().clamp_min(
        torch.finfo(expected.real.dtype).tiny
    )
    metrics = torch.tensor(
        [
            (difference.norm() / denominator).item(),
            difference.abs().max().item(),
        ],
        dtype=torch.float64,
        device=actual.device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    return metrics[0].item(), metrics[1].item()


class TestDistributedModel(DistributedTestCase):
    """End-to-end serial equivalence for the model selected by TEST_MODEL."""

    @parameterized.expand(CASES[MODEL])
    def test_distributed_model(
        self,
        _case_name,
        batch_per_dp,
        height,
        width,
        patch_size,
        channels,
        embed,
        heads,
        depth,
        tolerance,
    ):
        batch = batch_per_dp * comm.get_size("dp")
        reference, model, shard = build_models(
            MODEL, height, width, patch_size, channels, embed, heads, depth
        )
        reference = reference.to(self.device)
        model = model.to(self.device)

        # Rank zero defines the reference weights; metadata selects every local shard.
        with torch.no_grad():
            for parameter in reference.parameters():
                dist.broadcast(parameter, src=0)
        copy_reference_parameters(reference, model, shard)
        init_params_for_shared_weights(model)

        torch.manual_seed(5678)
        inp = torch.randn(
            batch, channels, height, width, device=self.device, requires_grad=True
        )
        target = torch.randn_like(inp)
        output = reference(inp)
        (output - target).square().mean().backward()

        device_ids = [self.device.index] if self.device.type == "cuda" else None
        ddp_model = init_ddp_model_and_reduction_hooks(
            model,
            device_ids=device_ids,
            output_device=self.device.index if device_ids else None,
        )
        with torch.no_grad():
            local_inp = split_forward_allgather_backward(
                inp.detach(), 0, None, "dp"
            )
            local_target = split_forward_allgather_backward(
                target, 0, None, "dp"
            )
            h_slice, w_slice = shard.core_slices
            local_inp = local_inp[..., h_slice, w_slice].clone()
            local_target = local_target[..., h_slice, w_slice]
        local_inp.requires_grad_(True)

        local_output = ddp_model(local_inp)
        mse_loss(local_output, local_target, (height, width)).backward()

        gathered_output = gather_field(local_output.detach(), shard)
        output_relative, output_absolute = error_metrics(gathered_output, output)
        if comm.get_world_rank() == 0:
            print(
                f"{MODEL}/{_case_name} output: "
                f"relative L2={output_relative:.6e}, "
                f"max absolute={output_absolute:.6e}"
            )
        self.assertLessEqual(
            output_relative,
            tolerance,
            f"output relative L2 error {output_relative:.6e} exceeds {tolerance:.6e}",
        )

        gathered_input_grad = (
            gather_field(local_inp.grad, shard) / comm.get_size("dp")
        )
        input_relative, input_absolute = error_metrics(
            gathered_input_grad, inp.grad
        )
        if comm.get_world_rank() == 0:
            print(
                f"{MODEL}/{_case_name} input gradients: "
                f"relative L2={input_relative:.6e}, "
                f"max absolute={input_absolute:.6e}"
            )
        self.assertLessEqual(
            input_relative,
            tolerance,
            f"input-gradient relative L2 error {input_relative:.6e} "
            f"exceeds {tolerance:.6e}",
        )

        assert_reference_gradients(
            self,
            reference,
            ddp_model.module,
            shard,
            tolerance=tolerance,
            print_summary=comm.get_world_rank() == 0,
        )


if __name__ == "__main__":
    unittest.main()
