import datetime as dt
import os

import pytest
import torch
import torch.distributed as dist

from distributed.helpers import init_params_for_shared_weights
from distributed.mappings import (
    allgather_forward_split_backward,
    init_ddp_model_and_reduction_hooks,
    roll_forward_reverseroll_backward,
)
from distributed.spatial_sharding import make_swin_spatial_shard
from networks.swin import SwinTransformer
from networks.swin_mp import DistributedSwinTransformer
from utils import comm
from utils.loss import mse_loss


@pytest.fixture(scope="module")
def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        pytest.skip("launch with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    tp = int(os.environ.get("TP", "1"))
    sp1 = int(os.environ.get("SP1", "1"))
    sp2 = int(os.environ.get("SP2", "1"))
    dp = world_size // (tp * sp1 * sp2)
    assert world_size == dp * tp * sp1 * sp2

    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        backend = "gloo"
    store = dist.TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=int(os.environ["MASTER_PORT"]),
        world_size=world_size,
        is_master=rank == 0,
        timeout=dt.timedelta(seconds=900),
    )
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        store=store,
    )
    comm.init_model_parallel_info(
        dp=dp,
        tp=tp,
        sp1=sp1,
        sp2=sp2,
        order="tp-sp1-sp2-dp",
    )
    yield device
    dist.destroy_process_group()


def _gather_spatial(x, shard):
    x = allgather_forward_split_backward(
        x, -1, shard.w_splits, "sp2"
    )
    return allgather_forward_split_backward(
        x, -2, shard.h_splits, "sp1"
    )


def _local_parameter_slice(parameter, metadata, shard, patch_size):
    slices = [slice(None)] * parameter.ndim
    for dim, group_name in enumerate(metadata["sharded"]):
        if group_name == "tp":
            width = parameter.shape[dim] // comm.get_size("tp")
            start = comm.get_rank("tp") * width
            slices[dim] = slice(start, start + width)
        elif group_name == "sp1":
            start = sum(shard.h_splits[: shard.h_rank]) // patch_size
            width = shard.h_splits[shard.h_rank] // patch_size
            slices[dim] = slice(start, start + width)
        elif group_name == "sp2":
            start = sum(shard.w_splits[: shard.w_rank]) // patch_size
            width = shard.w_splits[shard.w_rank] // patch_size
            slices[dim] = slice(start, start + width)
    return parameter[tuple(slices)]


def _copy_reference_weights(reference, model, shard):
    reference_parameters = dict(reference.named_parameters())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(
                _local_parameter_slice(
                    reference_parameters[name],
                    parameter.comm_metadata,
                    shard,
                    model.patch_size,
                )
            )


def test_distributed_roll_matches_global_roll(distributed_context):
    device = distributed_context
    height = 8 * comm.get_size("sp1")
    width = 8 * comm.get_size("sp2")
    global_x = torch.arange(
        height * width, dtype=torch.float32, device=device
    ).reshape(1, height, width, 1)
    h0 = 8 * comm.get_rank("sp1")
    w0 = 8 * comm.get_rank("sp2")
    local_x = (
        global_x[:, h0 : h0 + 8, w0 : w0 + 8]
        .clone()
        .requires_grad_(True)
    )
    output = roll_forward_reverseroll_backward(
        local_x, (-2, 3), (1, 2), ("sp1", "sp2")
    )
    expected = torch.roll(global_x, (-2, 3), (1, 2))[
        :, h0 : h0 + 8, w0 : w0 + 8
    ]
    torch.testing.assert_close(output, expected)

    output.square().sum().backward()
    expected_grad = 2 * global_x[:, h0 : h0 + 8, w0 : w0 + 8]
    torch.testing.assert_close(local_x.grad, expected_grad)


def test_swin_matches_reference(distributed_context):
    device = distributed_context
    torch.manual_seed(1234)
    dp_size = comm.get_size("dp")
    batch = 2 * dp_size
    image_size = (16, 16)
    patch_size = 2
    window_size = (4, 4)
    model_kwargs = dict(
        img_size=image_size,
        patch_size=patch_size,
        window_size=window_size,
        in_chans=2,
        out_chans=2,
        embed_dim=16,
        depth=2,
        num_heads=4,
        qkv_bias=True,
    )
    reference = SwinTransformer(**model_kwargs).to(device)
    shard = make_swin_spatial_shard(
        image_size,
        patch_size,
        window_size,
        comm.get_size("sp1"),
        comm.get_rank("sp1"),
        comm.get_size("sp2"),
        comm.get_rank("sp2"),
        "tp-sp1-sp2-dp",
    )
    model = DistributedSwinTransformer(
        **model_kwargs,
        spatial_shard=shard,
    ).to(device)
    _copy_reference_weights(reference, model, shard)
    init_params_for_shared_weights(model)
    model = init_ddp_model_and_reduction_hooks(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
    )

    torch.manual_seed(5678)
    x = torch.randn(batch, 2, *image_size, device=device, requires_grad=True)
    target = torch.randn_like(x)
    reference_output = reference(x)
    (reference_output - target).square().mean().backward()

    batch_width = batch // dp_size
    batch_start = comm.get_rank("dp") * batch_width
    h_slice, w_slice = shard.core_slices
    local_x = (
        x.detach()[
            batch_start : batch_start + batch_width,
            :,
            h_slice,
            w_slice,
        ]
        .clone()
        .requires_grad_(True)
    )
    local_target = target.detach()[
        batch_start : batch_start + batch_width,
        :,
        h_slice,
        w_slice,
    ]
    local_output = model(local_x)
    mse_loss(local_output, local_target, image_size).backward()

    output = _gather_spatial(local_output.detach(), shard)
    output = allgather_forward_split_backward(
        output, 0, None, "dp"
    )
    torch.testing.assert_close(output, reference_output, rtol=2e-4, atol=2e-5)

    input_grad = _gather_spatial(local_x.grad, shard)
    input_grad = allgather_forward_split_backward(
        input_grad, 0, None, "dp"
    ) / dp_size
    torch.testing.assert_close(input_grad, x.grad, rtol=5e-4, atol=2e-5)

    distributed_model = model.module
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in distributed_model.named_parameters():
        expected_grad = _local_parameter_slice(
            reference_parameters[name].grad,
            parameter.comm_metadata,
            shard,
            patch_size,
        )
        torch.testing.assert_close(
            parameter.grad,
            expected_grad,
            rtol=1e-3,
            atol=2e-5,
            msg=lambda message, name=name: f"{name}: {message}",
        )
