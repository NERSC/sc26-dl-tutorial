import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from parameterized import parameterized

from utils import comm
from tests.helpers import DistributedTestCase
from distributed.spatial_sharding import make_grid_spatial_shard
from utils.loss import mse_loss
from utils.metrics import rmse, rmse_spatial
from utils.data_loader import SynDataset
from utils.dali_es_helper import SynES


def grid_shard(shape):
    """this rank's shard of a plain (H, W) grid, as the trainer builds it for the FNO"""
    return make_grid_spatial_shard(
        shape,
        sp1_size=comm.get_size("sp1"),
        sp1_rank=comm.get_rank("sp1"),
        sp2_size=comm.get_size("sp2"),
        sp2_rank=comm.get_rank("sp2"),
    )


class Params:
    """the few config keys the data loaders read"""

    def __init__(self, data_path, img_size, read_mode="local"):
        self.dt = 1
        self.n_in_channels = 4
        self.n_out_channels = 4
        self.img_size = img_size
        self.limit_nsamples = None
        self.limit_nsamples_val = None
        self.valid_files = ["0003"]
        self.data_path = data_path
        self.stats_path = str(Path(data_path) / "stats.h5")
        self.spatial_read_mode = read_mode

    def get(self, key, default=None):
        return getattr(self, key, default)


class TestShardedIO(DistributedTestCase):
    """The FNO on PR #3's sharded I/O: grid shards, local loader reads, and the shard-aware loss"""

    @parameterized.expand([[(32, 64)], [(31, 62)]])
    def test_loss_and_metric_match_full(self, shape):
        h, w = shape
        pred = torch.randn(3, 4, h, w, device=self.device)
        tar = torch.randn(3, 4, h, w, device=self.device)
        dist.broadcast(pred, src=0)
        dist.broadcast(tar, src=0)
        shard = grid_shard(shape)

        # values: the full MSE and RMSE, from the shards
        pred_ref = pred.clone().requires_grad_(True)
        loss_ref = (pred_ref - tar).square().mean()
        p_loc = self._split(pred).clone().requires_grad_(True)
        loss = mse_loss(p_loc, self._split(tar), shard.global_shape)
        torch.testing.assert_close(loss, loss_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            rmse_spatial(self._split(pred), self._split(tar), shard), rmse(pred, tar), atol=1e-5, rtol=1e-5
        )

        # gradient: each rank gets the gradient with respect to its own shard, without double counting
        loss_ref.backward()
        loss.backward()
        torch.testing.assert_close(p_loc.grad, self._split(pred_ref.grad), atol=1e-7, rtol=1e-5)

    def test_grid_shard_matches_the_split_helper(self):
        """the shard object and the test helper split the grid the same way, so the loader tile
        is the shard the model and the FFT transposes expect"""
        for shape in [(32, 64), (31, 62)]:
            shard = grid_shard(shape)
            full = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(1, 1, *shape)
            h_slice, w_slice = shard.core_slices
            torch.testing.assert_close(full[..., h_slice, w_slice], self._split(full))
            self.assertEqual(shard.local_shape, tuple(self._split(full).shape[-2:]))

    @parameterized.expand([[(32, 64), "local"], [(31, 62), "local"], [(31, 62), "all"]])
    def test_loaders_load_the_tile(self, img_size, read_mode):
        """both loaders return exactly this rank's tile of the sample the full loader returns"""
        holder = [None]
        if dist.get_rank() == 0:
            data_dir = tempfile.mkdtemp()
            rng = np.random.default_rng(0)
            for f in range(4):
                with h5py.File(Path(data_dir) / f"trajectory_{f:04d}.h5", "w") as h:
                    h["fields"] = rng.standard_normal((5, 4, 40, 70)).astype(np.float32)
            with h5py.File(Path(data_dir) / "stats.h5", "w") as h:
                h["mean"] = np.zeros(4, dtype=np.float32)
                h["std"] = np.ones(4, dtype=np.float32)
            holder = [data_dir]
        dist.broadcast_object_list(holder, src=0)
        data_dir = holder[0]

        shard = grid_shard(img_size)
        h_slice, w_slice = shard.core_slices
        params = Params(data_dir, img_size, read_mode)
        full = SynDataset(params, data_dir, train=True)
        tile = SynDataset(params, data_dir, train=True, spatial_shard=shard)
        for idx in (0, 7, len(full) - 1):
            for a, b in zip(full[idx], tile[idx]):
                self.assertEqual(tuple(b.shape), (4, *shard.local_shape))
                torch.testing.assert_close(b, a[:, h_slice, w_slice])
            # the tiles of all spatial ranks tile the image
            inp = self._gather(tile[idx][0].to(self.device)[None], *img_size)[0].cpu()
            torch.testing.assert_close(inp, full[idx][0])

        # the DALI external source, without DALI: it is plain numpy/h5py
        class Info:
            def __init__(self, i):
                self.iteration = self.idx_in_epoch = i
                self.epoch_idx = 0

        args = (data_dir, False, 1, 1, img_size, 4, 4, 1, 0, None)
        es_full = SynES(*args, enable_logging=False)
        es_tile = SynES(*args, spatial_shard=shard, spatial_read_mode=read_mode, enable_logging=False)
        for i in (0, 3):
            for a, b in zip(es_full(Info(i)), es_tile(Info(i))):
                np.testing.assert_array_equal(b, a[:, h_slice, w_slice])
