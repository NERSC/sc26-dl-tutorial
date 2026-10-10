import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils import comm
from distributed.helpers import (
    _gather,
    _reduce,
    _reduce_scatter,
    _split,
    compute_split_shapes,
    validate_param_comm_metadata,
)


class _IdentityForwardAllReduceBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, comm_name_):
        return input_

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.comm_name = inputs

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.comm_name), None


class _AllReduceForwardIdentityBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, comm_name_):
        return _reduce(input_, comm_name_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _AllGatherForwardSplitBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, dim_, shapes_, comm_name_):
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _split(grad_output, ctx.dim, ctx.comm_name, ctx.shapes),
            None,
            None,
            None,
        )


class _SplitForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, dim_, shapes_, comm_name_):
        return _split(input_, dim_, comm_name_, shapes_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input_, ctx.dim, shapes, ctx.comm_name = inputs
        ctx.shapes = shapes or compute_split_shapes(
            input_.shape[ctx.dim], comm.get_size(ctx.comm_name)
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


class _ReduceScatterForwardAllGatherBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, dim_, shapes_, comm_name_):
        return _reduce_scatter(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input_, ctx.dim, shapes, ctx.comm_name = inputs
        ctx.shapes = shapes or compute_split_shapes(
            input_.shape[ctx.dim], comm.get_size(ctx.comm_name)
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(grad_output, ctx.dim, ctx.shapes, ctx.comm_name),
            None,
            None,
            None,
        )


class _AllGatherForwardReduceScatterBackward(torch.autograd.Function):
    @staticmethod
    def forward(input_, dim_, shapes_, comm_name_):
        return _gather(input_, dim_, shapes_, comm_name_)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, ctx.dim, ctx.shapes, ctx.comm_name = inputs

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _reduce_scatter(
                grad_output, ctx.dim, ctx.shapes, ctx.comm_name
            ),
            None,
            None,
            None,
        )


@torch.compiler.disable
def identity_forward_allreduce_backward(input_, comm_name):
    return _IdentityForwardAllReduceBackward.apply(input_, comm_name)


@torch.compiler.disable
def allreduce_forward_identity_backward(input_, comm_name):
    return _AllReduceForwardIdentityBackward.apply(input_, comm_name)


@torch.compiler.disable
def allgather_forward_split_backward(input_, dim, shapes, comm_name):
    return _AllGatherForwardSplitBackward.apply(
        input_, dim, shapes, comm_name
    )


@torch.compiler.disable
def split_forward_allgather_backward(
    input_, dim, shapes=None, comm_name=None
):
    if comm_name is None:
        raise ValueError("comm_name is required")
    return _SplitForwardAllGatherBackward.apply(
        input_, dim, shapes, comm_name
    )


@torch.compiler.disable
def reducescatter_forward_allgather_backward(
    input_, dim, shapes, comm_name
):
    return _ReduceScatterForwardAllGatherBackward.apply(
        input_, dim, shapes, comm_name
    )


@torch.compiler.disable
def allgather_forward_reducescatter_backward(
    input_, dim, shapes, comm_name
):
    return _AllGatherForwardReduceScatterBackward.apply(
        input_, dim, shapes, comm_name
    )


def _completed_future(value):
    future = torch.futures.Future()
    future.set_result(value)
    return future


def _coalesce_gradients(gradients):
    return torch.cat([gradient.contiguous().view(-1) for gradient in gradients])


def _copy_coalesced_gradients(coalesced, gradients):
    offset = 0
    for gradient in gradients:
        next_offset = offset + gradient.numel()
        gradient.copy_(coalesced[offset:next_offset].view_as(gradient))
        offset = next_offset


def init_ddp_model_and_reduction_hooks(
    model,
    device_ids,
    output_device,
    bucket_cap_mb=25,
    find_unused_parameters=False,
    gradient_as_bucket_view=True,
    static_graph=False,
):
    """Wrap with DDP and consume all validated non-DP reduction metadata."""
    if not dist.is_initialized():
        return model

    validate_param_comm_metadata(model)
    group_order = [
        name
        for name in comm.get_names()
        if name != "dp" and comm.get_size(name) > 1
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    signatures = [
        tuple(
            group_name
            for group_name in group_order
            if group_name in parameter.comm_metadata["reduce"]
        )
        for parameter in parameters
    ]

    dp_group = comm.get_group("dp")
    backend = dist.get_backend(dp_group)
    if backend == "gloo":
        device_ids = None
        output_device = None

    ddp_model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=output_device,
        bucket_cap_mb=bucket_cap_mb,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
        process_group=dp_group,
    )

    if not any(signatures):
        return ddp_model

    unique_signatures = set(signatures)
    if len(unique_signatures) == 1:
        signature = next(iter(unique_signatures))
        if len(signature) == 1:
            reduce_name = signature[0]
            fused_name = (
                reduce_name
                if comm.get_size("dp") == 1
                else f"dp-{reduce_name}"
            )
            if not comm.is_initialized(fused_name):
                raise ValueError(
                    f"Required composite reduction group {fused_name!r} is not initialized"
                )

            def fused_reduction_hook(
                state: object, bucket: dist.GradBucket
            ) -> torch.futures.Future[torch.Tensor]:
                buffer = bucket.buffer()
                if comm.get_size(fused_name) == 1:
                    return _completed_future(buffer)
                future = dist.all_reduce(
                    buffer,
                    op=dist.ReduceOp.SUM,
                    group=comm.get_group(fused_name),
                    async_op=True,
                ).get_future()
                return future.then(
                    lambda completed: completed.value()[0].div_(
                        comm.get_size("dp")
                    )
                )

            ddp_model.register_comm_hook(None, fused_reduction_hook)
            return ddp_model

    def reduction_comm_hook(
        state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        buffer = bucket.buffer()
        if comm.get_size("dp") > 1:
            future = dist.all_reduce(
                buffer,
                op=dist.ReduceOp.SUM,
                group=dp_group,
                async_op=True,
            ).get_future()
            future = future.then(
                lambda completed: completed.value()[0].div_(
                    comm.get_size("dp")
                )
            )
        else:
            future = _completed_future(buffer)

        bucket_parameters = bucket.parameters()
        bucket_gradients = bucket.gradients()
        for group_name in group_order:
            selected = [
                gradient
                for parameter, gradient in zip(
                    bucket_parameters, bucket_gradients
                )
                if group_name in parameter.comm_metadata["reduce"]
            ]
            if not selected:
                continue

            def reduce_selected(completed, gradients=selected, group=group_name):
                coalesced = _coalesce_gradients(gradients)
                dist.all_reduce(
                    coalesced,
                    op=dist.ReduceOp.SUM,
                    group=comm.get_group(group),
                )
                _copy_coalesced_gradients(coalesced, gradients)
                return bucket.buffer()

            future = future.then(reduce_selected)

        return future

    ddp_model.register_comm_hook(None, reduction_comm_hook)
    return ddp_model
