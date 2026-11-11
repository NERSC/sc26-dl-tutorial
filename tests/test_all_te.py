"""Distributed equivalence test for the Transformer Engine ViT.

The reference is the same ``networks.vit_te.VisionTransformer`` built while the
model-parallel groups are hidden, so Transformer Engine constructs full
layers. The distributed model uses the live ``tp`` / ``sp1`` / ``sp2`` mesh.
Both forwards run under bfloat16 autocast.
"""

import unittest
from contextlib import contextmanager

import torch
import torch.distributed as dist
from parameterized import parameterized

from distributed.helpers import init_params_for_shared_weights
from distributed.mappings import (
    allgather_forward_split_backward,
    init_ddp_model_and_reduction_hooks,
    split_forward_allgather_backward,
)
from distributed.spatial_sharding import make_vit_spatial_shard
from networks.vit_te import VisionTransformer
from tests.helpers import (
    DistributedTestCase,
    assert_reference_gradients,
    copy_reference_parameters,
)
from utils import comm
from utils.loss import mse_loss


# Same grids as the ViT rows in tests/test_all.py. bfloat16 autocast needs the
# looser tolerance used by the sc25 Transformer Engine test.
# (name, batch/DP rank, H, W, patch, channels, embed, heads, depth, tolerance)
CASES = [
    ("small", 4, 128, 128, 8, 4, 512, 8, 2, 1e-2),
    ("rectangular", 4, 360, 720, 8, 20, 512, 8, 2, 1e-2),
]


def expand_cases():
    """Turn each case row into the arguments the test method receives."""
    expanded = []
    for name, batch, height, width, patch, channels, embed, heads, depth, tolerance in CASES:
        spec = dict(
            patch_size=patch,
            channels=channels,
            embed=embed,
            heads=heads,
            depth=depth,
        )
        expanded.append((name, batch, height, width, tolerance, spec))
    return expanded


@contextmanager
def _serial_transformer_engine():
    """Hide model-parallel groups so Transformer Engine builds unsharded layers."""
    saved = dict(comm._COMM_GROUPS)
    comm._COMM_GROUPS.clear()
    try:
        yield
    finally:
        comm._COMM_GROUPS.clear()
        comm._COMM_GROUPS.update(saved)


def build_models(height, width, spec):
    """Build an unsharded reference and this rank's spatially sharded ViT."""
    kwargs = dict(
        img_size=(height, width),
        patch_size=spec["patch_size"],
        in_chans=spec["channels"],
        out_chans=spec["channels"],
        embed_dim=spec["embed"],
        depth=spec["depth"],
        num_heads=spec["heads"],
        qkv_bias=True,
    )
    full_shard = make_vit_spatial_shard(
        (height, width), spec["patch_size"], 1, 0, 1, 0, "tp-sp1-sp2-dp"
    )
    shard = make_vit_spatial_shard(
        (height, width),
        spec["patch_size"],
        comm.get_size("sp1"),
        comm.get_rank("sp1"),
        comm.get_size("sp2"),
        comm.get_rank("sp2"),
        "tp-sp1-sp2-dp",
    )
    with _serial_transformer_engine():
        reference = VisionTransformer(**kwargs, spatial_shard=full_shard)
    model = VisionTransformer(**kwargs, spatial_shard=shard)
    return reference, model, shard


def gather_field(tensor, shard):
    """Gather a spatially sharded field back to the full batch and grid."""
    tensor = allgather_forward_split_backward(
        tensor, -1, shard.w_splits, "sp2"
    )
    tensor = allgather_forward_split_backward(
        tensor, -2, shard.h_splits, "sp1"
    )
    return allgather_forward_split_backward(tensor, 0, None, "dp")


def error_metrics(actual, expected):
    """Return the worst relative L2 and absolute errors across all ranks."""
    actual = actual.detach().float()
    expected = expected.detach().float()
    difference = actual - expected
    denominator = expected.norm().clamp_min(torch.finfo(expected.dtype).tiny)
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


@unittest.skipUnless(
    torch.cuda.is_available(), "Transformer Engine tests require CUDA"
)
class TestTransformerEngineViT(DistributedTestCase):
    """End-to-end equivalence of the unsharded and sharded Transformer Engine ViTs."""

    @parameterized.expand(expand_cases())
    def test_distributed_model(
        self,
        _case_name,
        batch_per_dp,
        height,
        width,
        tolerance,
        spec,
    ):
        channels = spec["channels"]
        batch = batch_per_dp * comm.get_size("dp")
        reference, model, shard = build_models(height, width, spec)
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
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            output = reference(inp)
        (output.float() - target).square().mean().backward()

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

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            local_output = ddp_model(local_inp)
        mse_loss(local_output.float(), local_target, (height, width)).backward()

        gathered_output = gather_field(local_output.detach(), shard)
        output_relative, output_absolute = error_metrics(gathered_output, output)
        if comm.get_world_rank() == 0:
            print(
                f"vit-te/{_case_name} output: "
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
                f"vit-te/{_case_name} input gradients: "
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
