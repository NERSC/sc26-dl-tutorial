import torch
import os
import datetime as dt
import unittest

import torch.distributed as dist

from distributed.helpers import compute_split_shapes, _split
from utils import comm

def copy_mlp_weights(mlp_layer, mlp_layer_distributed):
    """copy the weights of mlp into the correct shard of mlp_dist"""
    tp = comm.get_size("tp")
    rank_tp = comm.get_rank("tp")  # which tp rank
    embed_local = mlp_layer.fc1.weight.shape[0] // tp
    start = rank_tp * embed_local
    end = start + embed_local

    if embed_local == mlp_layer_distributed.fc1.weight.shape[0]:
        # model is sharded
        if comm.get_world_rank() == 0:
            print("copying weights for sharded model")
        with torch.no_grad():
            mlp_layer_distributed.fc1.weight.copy_(mlp_layer.fc1.weight[start:end, :])
            mlp_layer_distributed.fc2.weight.copy_(mlp_layer.fc2.weight[:, start:end])


class DistributedTestCase(unittest.TestCase):
    """Sets up a (dp, tp, sp1, sp2) process grid and helpers for (H, W)-sharded tensors."""


    @classmethod
    def setUpClass(cls):
        cls.world_size = int(os.getenv("WORLD_SIZE", 1))
        cls.world_rank = int(os.getenv("RANK", 0))
        tp = int(os.getenv("TP", 1))
        sp1 = int(os.getenv("SP1", 1))
        sp2 = int(os.getenv("SP2", 1))
        assert cls.world_size > 1, "Running distributed tests on single GPU"
        assert cls.world_size % (tp * sp1 * sp2) == 0
        dp = cls.world_size // (tp * sp1 * sp2)

        if torch.cuda.is_available():
            cls.device = torch.device(f"cuda:{cls.world_rank % torch.cuda.device_count()}")
            torch.cuda.set_device(cls.device)
            backend = "nccl"
        else:
            cls.device = torch.device("cpu")
            backend = "gloo"
        torch.manual_seed(333)

        store = dist.TCPStore(
            host_name=os.getenv("MASTER_ADDR"),
            port=int(os.getenv("MASTER_PORT", 0)),
            world_size=cls.world_size,
            is_master=(cls.world_rank == 0),
            timeout=dt.timedelta(seconds=900),
        )
        dist.init_process_group(
            backend=backend, rank=cls.world_rank, world_size=cls.world_size, store=store
        )
        comm.init_model_parallel_info(dp=dp, tp=tp, sp1=sp1, sp2=sp2, order="tp-sp1-sp2-dp")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group(None)

    def _split(self, x, h_dim=2, w_dim=3):
        """take this rank's (H, W) shard; defaults to channels-first (B, C, H, W)"""
        x = _split(x, h_dim, "sp1")
        return _split(x, w_dim, "sp2")

    def _gather_dim(self, x, dim, full, comm_name):
        """all-gather with uneven shards (pad to the largest, gather, trim)"""
        sizes = compute_split_shapes(full, comm.get_size(comm_name))
        pad = max(sizes) - x.shape[dim]
        if pad > 0:
            shape = list(x.shape)
            shape[dim] = pad
            x = torch.cat([x, x.new_zeros(shape)], dim=dim)
        chunks = [torch.empty_like(x) for _ in sizes]
        dist.all_gather(chunks, x.contiguous(), group=comm.get_group(comm_name))
        return torch.cat([c.narrow(dim, 0, n) for c, n in zip(chunks, sizes)], dim=dim)

    def _gather(self, x, nh, nw, h_dim=2, w_dim=3):
        """gather an (H, W)-sharded tensor with global sizes nh, nw"""
        x = self._gather_dim(x, h_dim, nh, "sp1")
        return self._gather_dim(x, w_dim, nw, "sp2")
