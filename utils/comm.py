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
    """Returns the names of all available communicators."""
    return _COMM_GROUPS.keys()


def is_initialized(comm_name):
    """check if initialized."""
    return comm_name in _COMM_GROUPS


def get_group(comm_name):
    """Returns the group of a specified communicator."""
    if not is_initialized(comm_name):
        raise IndexError(f"Error, comm {comm_name} not initialized.")
    return _COMM_GROUPS[comm_name]


def get_size(comm_name):
    """Returns the size of a specified communicator."""
    if (not dist.is_initialized()) or (not is_initialized(comm_name)):
        return 1
    else:
        return dist.get_world_size(group=get_group(comm_name))


def get_rank(comm_name):
    """Returns the rank in a specified communicator."""
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
    """Returns the world size"""
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def get_world_rank():
    """Returns the world rank"""
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_local_rank():
    """Returns the local rank of the current process."""
    if not dist.is_initialized():
        return 0
    else:
        if os.getenv("LOCAL_RANK") is not None:
            # Use env var if available
            return int(os.getenv("LOCAL_RANK"))
        else:
            return get_world_rank() % torch.cuda.device_count()


def init(params, verbose=False):
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
    """Initial torch distributed process group
    Uses NCCL
    """
    world_size = int(os.getenv("WORLD_SIZE", 1))
    world_rank = int(os.getenv("RANK", 0))
    port = int(os.getenv("MASTER_PORT", 0))
    master_address = os.getenv("MASTER_ADDR")
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    with disable_logging():
        # create tcp store
        store = dist.TCPStore(
            host_name=master_address,
            port=port,
            world_size=world_size,
            is_master=(world_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )

        # initialize process groups
        dist.init_process_group(
            backend="nccl", rank=world_rank, world_size=world_size, store=store
        )


def init_model_parallel_info(
    dp=1, tp=1, sp1=1, sp2=1, order="tp-sp1-sp2-dp", verbose=False
):

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
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        data, tensor, and two-dimensional spatial parallelism.
        """
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
