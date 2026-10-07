import torch
import torch.distributed as dist
from utils import comm


def validate_param_comm_metadata(model):
    """Validate communication metadata for every trainable parameter."""
    valid_groups = set(comm.get_names())
    required_keys = {"sharded", "shared", "reduce"}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not hasattr(param, "comm_metadata"):
            raise ValueError(f"{name}: missing comm_metadata")
        metadata = param.comm_metadata
        if set(metadata) != required_keys:
            raise ValueError(
                f"{name}: comm_metadata keys must be {sorted(required_keys)}, "
                f"got {sorted(metadata)}"
            )
        if not all(isinstance(metadata[key], list) for key in required_keys):
            raise TypeError(f"{name}: all comm_metadata values must be lists")
        if len(metadata["sharded"]) != param.ndim:
            raise ValueError(
                f"{name}: sharded metadata has {len(metadata['sharded'])} entries "
                f"for a rank-{param.ndim} parameter"
            )
        for group_name in metadata["sharded"]:
            if group_name is not None and group_name not in valid_groups:
                raise ValueError(f"{name}: unknown sharded group {group_name!r}")
        for field in ("shared", "reduce"):
            if any(group_name is None for group_name in metadata[field]):
                raise ValueError(f"{name}: None is only valid in sharded metadata")
            unknown = set(metadata[field]) - valid_groups
            if unknown:
                raise ValueError(f"{name}: unknown {field} groups {sorted(unknown)}")


def init_params_for_shared_weights(model):
    """Synchronize replicated model-parallel parameters from group roots."""
    validate_param_comm_metadata(model)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            for comm_group in param.comm_metadata["shared"]:
                if comm.get_size(comm_group) > 1:
                    dist.broadcast(
                        param,
                        src=comm.get_root(comm_group),
                        group=comm.get_group(comm_group),
                    )


# distributed primitives
# helper routine to compute uneven splitting in balanced way:
def compute_split_shapes(size, num_chunks):
    # treat trivial case first
    if num_chunks == 1:
        return [size]

    # first, check if we can split using div-up to balance the load:
    chunk_size = (size + num_chunks - 1) // num_chunks
    last_chunk_size = max(0, size - chunk_size * (num_chunks - 1))
    if last_chunk_size == 0:
        # in this case, the last shard would be empty, split with floor instead:
        chunk_size = size // num_chunks
        last_chunk_size = size - chunk_size * (num_chunks - 1)

    # generate sections list
    sections = [chunk_size for _ in range(num_chunks - 1)] + [last_chunk_size]

    return sections


def _reduce(input_, comm_name):
    """All-reduce the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm.get_size(comm_name) == 1:
        return input_

    output = input_.contiguous().clone()
    dist.all_reduce(output, group=comm.get_group(comm_name))
    return output


def split_tensor_along_dim(tensor, dim, num_chunks, sections=None):
    """Helper routine to split a tensor along a given dimension"""
    if dim < 0:
        dim += tensor.dim()
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(
            f"Error: Scattering along {dim} for a tensor of size {tensor.dim()}"
        )
    if tensor.shape[dim] < num_chunks and sections is None:
        raise ValueError(
            f"Error, cannot split dim {dim} of size {tensor.shape[dim]} into {num_chunks} chunks"
        )

    if sections is None:
        sections = compute_split_shapes(tensor.shape[dim], num_chunks)
    if len(sections) != num_chunks or sum(sections) != tensor.shape[dim]:
        raise ValueError(
            f"Invalid sections {sections} for dim {dim} of size {tensor.shape[dim]}"
        )
    return list(torch.split(tensor, sections, dim=dim))


def _split(input_, dim_, comm_name, sections=None):
    """Split the tensor along dim."""
    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    comm_size = comm.get_size(comm_name)
    if comm_size == 1:
        return input_

    # Split along  dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size, sections)

    # Note: torch.split does not create contiguous tensors by default.
    comm_rank = comm.get_rank(comm_name)
    output = input_list[comm_rank].contiguous()

    return output


def _gather(input_, dim_, shapes_, comm_name):
    """
    Gather tensors and concatinate along the dimension dim_.
    """
    comm_size = comm.get_size(comm_name)
    if (shapes_ is not None) and (len(shapes_) != comm_size):
        raise ValueError(f"Error: passed shapes of size not equal to {comm_size}")
    if dim_ < 0:
        dim_ += input_.dim()
    if dim_ < 0 or dim_ >= input_.dim():
        raise ValueError(
            f"Error: Gathering along {dim_} for a tensor of rank {input_.dim()}"
        )

    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm_size == 1:
        return input_

    input_ = input_.contiguous()
    input_shape = list(input_.shape)
    if shapes_ is not None:
        input_list = []
        for src in range(comm_size):
            input_shape[dim_] = shapes_[src]
            input_list.append(
                torch.empty(input_shape, dtype=input_.dtype, device=input_.device)
            )
    else:
        # assume equal shape on all ranks
        input_list = [torch.empty_like(input_) for _ in range(comm_size)]

    group = comm.get_group(comm_name)
    if shapes_ is not None and len(set(shapes_)) > 1 and dist.get_backend(group) == "gloo":
        # gloo's all_gather needs equal shapes: pad to the largest shard, gather, trim
        pad = max(shapes_) - input_.shape[dim_]
        if pad > 0:
            pad_shape = list(input_.shape)
            pad_shape[dim_] = pad
            input_ = torch.cat([input_, input_.new_zeros(pad_shape)], dim=dim_)
        padded = [torch.empty_like(input_) for _ in range(comm_size)]
        dist.all_gather(padded, input_, group=group)
        input_list = [t.narrow(dim_, 0, n) for t, n in zip(padded, shapes_)]
    else:
        dist.all_gather(input_list, input_, group=group)
    output = torch.cat(input_list, dim=dim_).contiguous()

    return output

def _reduce_scatter(input_, dim_, shapes_, comm_name):
    """
    Reduces and scatters along dim_
    """
    comm_size = comm.get_size(comm_name)
    if dim_ < 0:
        dim_ += input_.dim()
    if dim_ < 0 or dim_ >= input_.dim():
        raise ValueError(
            f"Error: Reduce-scatter along {dim_} for a tensor of rank {input_.dim()}"
        )

    # Bypass the function if we are using only 1 GPU or if
    # communicator is not initialized
    if comm_size == 1:
        return input_

    if shapes_ is None:
        shapes_ = compute_split_shapes(input_.shape[dim_], comm_size)
    if len(shapes_) != comm_size or sum(shapes_) != input_.shape[dim_]:
        raise ValueError(
            f"Invalid reduce-scatter shapes {shapes_} for dim size {input_.shape[dim_]}"
        )

    input_ = input_.contiguous()
    input_list = [
        t.contiguous()
        for t in split_tensor_along_dim(input_, dim_, comm_size, shapes_)
    ]
    comm_rank = comm.get_rank(comm_name)
    if len(set(shapes_)) == 1:
        output = torch.empty_like(input_list[comm_rank])
        dist.reduce_scatter(output, input_list, group=comm.get_group(comm_name))
        return output

    reduced = input_.clone()
    dist.all_reduce(reduced, group=comm.get_group(comm_name))
    return split_tensor_along_dim(
        reduced, dim_, comm_size, shapes_
    )[comm_rank].contiguous()


def _transpose(tensor, dim0, dim1, dim1_split_sizes, comm_name):
    """
    All-to-all transpose: dim0 (local in the input) becomes split across the group,
    dim1 (split in the input) becomes local. Returns the list of received chunks,
    to be concatenated along dim1, and the dim0 split sizes.
    """
    comm_size = comm.get_size(comm_name)
    if comm_size == 1:
        return [tensor], [tensor.size(dim0)]
    if len(dim1_split_sizes) != comm_size:
        raise ValueError(
            f"Error: passed shapes of size {len(dim1_split_sizes)} not equal to {comm_size}"
        )
    comm_rank = comm.get_rank(comm_name)

    # split and make send buffers contiguous
    x_send = [
        t.contiguous()
        for t in split_tensor_along_dim(tensor, dim0, comm_size)
    ]
    dim0_split_sizes = [x.shape[dim0] for x in x_send]

    # receive buffers: dim0 is the local chunk size, dim1 comes from each source
    x_shape = list(x_send[comm_rank].shape)
    x_recv = []
    for dim1_len in dim1_split_sizes:
        x_shape[dim1] = dim1_len
        x_recv.append(
            torch.empty(x_shape, dtype=tensor.dtype, device=tensor.device)
        )

    group = comm.get_group(comm_name)
    if dist.get_backend(group) == "gloo":
        # gloo's all_to_all only supports equal chunk sizes, use point-to-point instead
        x_recv[comm_rank].copy_(x_send[comm_rank])
        ops = []
        for peer in range(comm_size):
            if peer == comm_rank:
                continue
            global_peer = dist.get_global_rank(group, peer)
            ops.append(dist.P2POp(dist.isend, x_send[peer], global_peer, group))
            ops.append(dist.P2POp(dist.irecv, x_recv[peer], global_peer, group))
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    else:
        dist.all_to_all(x_recv, x_send, group=group)

    return x_recv, dim0_split_sizes
