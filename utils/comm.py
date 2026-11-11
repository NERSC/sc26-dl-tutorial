"""Process groups for data, tensor, and spatial parallelism."""

import os
import logging
from utils.logging_utils import disable_logging
from utils.rank_generator import RankGenerator
import torch
import math
import numpy as np
import torch.distributed as dist
import datetime as dt
from typing import Union

# dummy placeholder
_COMM_GROUPS = {}


# routines for specific comm groups
def get_names():
    """Return the names of the initialized communicators."""
    return _COMM_GROUPS.keys()


def is_initialized(comm_name):
    """Return whether ``comm_name`` has been created."""
    return comm_name in _COMM_GROUPS


def get_group(comm_name):
    """Return the process group named ``comm_name``."""
    if not is_initialized(comm_name):
        raise IndexError(f"Error, comm {comm_name} not initialized.")
    return _COMM_GROUPS[comm_name]


def get_size(comm_name):
    """Return the number of ranks in ``comm_name``, or 1 when it is absent."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 1
    else:
        return dist.get_world_size(group=get_group(comm_name))


def get_rank(comm_name):
    """Return this process's rank in ``comm_name``, or 0 when it is absent."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 0
    else:
        return dist.get_rank(group=get_group(comm_name))


def get_root(comm_name):
    """Return the global rank of rank zero in a communicator."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 0
    return dist.get_process_group_ranks(get_group(comm_name))[0]


# routines for world comms
def get_world_size():
    """Return the distributed world size, or 1 when distributed is not initialized."""
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def get_world_rank():
    """Return this process's global rank, or 0 when distributed is not initialized."""
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_local_rank():
    """Return this process's rank on its node."""
    if not dist.is_initialized():
        return 0
    else:
        if os.getenv("LOCAL_RANK") is not None:
            # Use env var if available
            return int(os.getenv("LOCAL_RANK"))
        else:
            return get_world_rank() % torch.cuda.device_count()


def init(params, verbose=False):
    """Initialize NCCL and the data, tensor, and spatial process groups."""
    # init torch.distributed
    init_process_group()

    # set model parallel sizes
    tp = params.get("tp", 1)
    sp1 = params.get("sp1", 1)
    sp2 = params.get("sp2", 1)
    model_parallel_size = tp * sp1 * sp2
    assert get_world_size() % model_parallel_size == 0, (
        f"ERROR: world size {get_world_size()} is not divisible by "
        f"tp*sp1*sp2={model_parallel_size}"
    )
    dp = get_world_size() // model_parallel_size
    assert dp >= 1, "ERROR: data parallel wireup failed since dp = {}".format(dp)
    logging.info(
        "Setting DP = {}, TP = {}, SP1 = {}, SP2 = {}".format(dp, tp, sp1, sp2)
    )

    # init model + dp groups individually
    init_model_parallel_info(
        dp=dp,
        tp=tp,
        sp1=sp1,
        sp2=sp2,
        order=params.get("order", "tp-sp1-sp2-dp"),
        verbose=verbose,
    )


def init_process_group():
    """Initialize the NCCL process group from the distributed environment."""
    # RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT come from torchrun (or from
    # export_DDP_vars.sh under a plain srun launch). env:// also reuses the
    # torchrun agent's TCP store instead of trying to bind a second one.
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    with disable_logging():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=dt.timedelta(seconds=900),
            device_id=device,
        )


def init_model_parallel_info(
    dp=1, tp=1, sp1=1, sp2=1, order="tp-sp1-sp2-dp", verbose=False
):
    """Create the orthogonal process groups for the requested mesh."""

    world_size = get_world_size()
    world_rank = get_world_rank()
    expected_world_size = dp * tp * sp1 * sp2
    if world_size != expected_world_size:
        raise ValueError(
            f"World size {world_size} does not match "
            f"dp*tp*sp1*sp2={expected_world_size}"
        )

    rank_gen = RankGenerator(
        dp=dp,
        tp=tp,
        sp1=sp1,
        sp2=sp2,
        order=order,
    )

    def generator_wrapper(group_type, **kwargs):
        """Yield the rank lists for one communicator name."""
        ranks = rank_gen.get_ranks(group_type, **kwargs)
        for x in ranks:
            yield x

    # build the different parallel groups
    global _COMM_GROUPS  # others need access to this
    groups_to_build = [
        "dp",
        "tp",
        "sp1",
        "sp2",
        "sp1-sp2",
        "tp-sp1-sp2",
        "dp-sp1-sp2",
    ]
    for grp in groups_to_build:
        for ranks in generator_wrapper(grp):
            group = dist.new_group(ranks)
            if world_rank in ranks:
                _COMM_GROUPS[grp] = group


def process_comm_list(input_list):
    """Given a list of comms, merge them
    Ex: ['tp', 'sp1'] is ['tp-sp1']
    """
    if not input_list or all(item is None for item in input_list):
        return []

    # filter out None values (ex: [None, 'tp] becomes ['tp'])
    filtered_list = [item for item in input_list if item is not None]

    if not filtered_list:
        return []
    elif len(filtered_list) == 1:
        return filtered_list
    else:
        return ["-".join(filtered_list)]
