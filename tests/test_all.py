"""Master distributed-equivalence test.

Run with TEST_MODEL set to vit, swin, fno, or convnext.
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
from networks.convnext import ConvNeXtField
from networks.convnext_mp import DistributedConvNeXt
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
    local_parameter_slice,
)
from utils import comm
from utils.loss import mse_loss


MODEL = os.getenv("TEST_MODEL", "vit").lower()
if MODEL not in {"vit", "swin", "fno", "convnext"}:
    raise ValueError("TEST_MODEL must be one of: vit, swin, fno, convnext")

CASES = {
    "convnext": [
        ("even", 2, 32, 48, 2, 2, 8, 1, 2, 2e-5),
        ("kernel3", 2, 14, 14, 2, 2, 8, 1, 2, 2e-5),
        ("pointwise", 2, 8, 8, 2, 2, 8, 1, 2, 2e-5),
        # 15x15 patches: uneven under two or four spatial ranks; min tile=3.
        ("uneven", 2, 30, 30, 2, 2, 8, 1, 2, 2e-5),
    ],
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


def build_models(model_name, height, width, patch_size, channels, embed, heads, depth, kernel_size=7):
    """Build matching serial/distributed models and this rank's spatial shard."""
    mesh = (
        comm.get_size("sp1"),
        comm.get_rank("sp1"),
        comm.get_size("sp2"),
        comm.get_rank("sp2"),
    )
    if model_name == "convnext":
        kwargs = dict(img_size=(height, width), patch_size=patch_size,
                      in_chans=channels, out_chans=channels, embed_dim=embed,
                      depth=depth, kernel_size=kernel_size, layer_scale_init=0.5)
        shard = make_vit_spatial_shard(
            (height, width), patch_size, *mesh, "tp-sp1-sp2-dp"
        )
        return (ConvNeXtField(**kwargs),
                DistributedConvNeXt(**kwargs, spatial_shard=shard), shard)
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
        kernel_size = {"kernel3": 3, "pointwise": 1}.get(_case_name, 7)
        amp_mode = os.getenv("TEST_AMP", "none") if MODEL == "convnext" else "none"
        self.assertIn(amp_mode, {"none", "bf16", "fp16"})
        amp_enabled = amp_mode != "none"
        amp_dtype = torch.float16 if amp_mode == "fp16" else torch.bfloat16
        if amp_enabled:
            tolerance = 0.05
        batch = batch_per_dp * comm.get_size("dp")
        reference, model, shard = build_models(
            MODEL, height, width, patch_size, channels, embed, heads, depth, kernel_size
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
        raw_model = model
        backend = os.getenv("TEST_COMPILE_BACKEND", "") if MODEL == "convnext" else ""
        if backend:
            if backend == "inductor" and self.device.type != "cuda":
                self.fail("Inductor acceptance must run on CUDA")
            if backend == "aot_eager":
                from torch._dynamo.backends.common import aot_autograd
                from functorch.compile import make_boxed_func
                from torch._functorch.partitioners import min_cut_rematerialization_partition

                # Match Inductor's partitioner to avoid replaying halo collectives.
                backend = aot_autograd(
                    fw_compiler=lambda graph, _: make_boxed_func(graph.forward),
                    bw_compiler=lambda graph, _: make_boxed_func(graph.forward),
                    partition_fn=min_cut_rematerialization_partition,
                )
            model = torch.compile(model, backend=backend)

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

        if MODEL == "convnext":
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.9)
            optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.05, momentum=0.9)
        # Repeated iterations exercise DDP bucket rebuilds and compiled backward.
        for step in range(3 if MODEL == "convnext" else 1):
            if MODEL == "convnext":
                reference_optimizer.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
            inp.grad = None
            local_inp.grad = None
            with torch.autocast(self.device.type, enabled=amp_enabled, dtype=amp_dtype):
                output = reference(inp)
            (output.float() - target).square().mean().backward()
            with torch.autocast(self.device.type, enabled=amp_enabled, dtype=amp_dtype):
                local_output = ddp_model(local_inp)
            if amp_enabled:
                self.assertEqual(local_output.dtype, amp_dtype)
            mse_loss(local_output.float(), local_target, (height, width)).backward()

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
                raw_model,
                shard,
                tolerance=tolerance,
                print_summary=comm.get_world_rank() == 0,
            )

            if MODEL == "convnext":
                reference_optimizer.step()
                optimizer.step()
                ref_parameters = dict(reference.named_parameters())
                for name, parameter in raw_model.named_parameters():
                    expected = local_parameter_slice(name, ref_parameters[name], parameter, shard)
                    torch.testing.assert_close(
                        parameter, expected,
                        atol=1e-4 if amp_enabled else 1e-6,
                        rtol=0.05 if amp_enabled else 2e-4,
                    )


if __name__ == "__main__":
    unittest.main()
