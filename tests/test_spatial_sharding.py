from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from utils.dali_es_helper import SynES
from utils.data_loader import SynDataset
from utils.loss import mse_loss
from utils.metrics import rmse, rmse_spatial
from distributed.helpers import compute_split_shapes
from distributed.spatial_sharding import (
    make_swin_spatial_shard,
    make_vit_spatial_shard,
)


class Params(SimpleNamespace):
    def get(self, name, default=None):
        return getattr(self, name, default)


def test_compute_split_shapes_is_balanced_and_nonempty():
    assert compute_split_shapes(10, 4) == [3, 3, 2, 2]
    assert compute_split_shapes(12, 4) == [3, 3, 3, 3]
    with pytest.raises(ValueError, match="every chunk must be non-empty"):
        compute_split_shapes(3, 4)
    with pytest.raises(ValueError, match="must be positive"):
        compute_split_shapes(3, 0)


def test_vit_policy_splits_patch_rows_and_preserves_token_order():
    shards = [
        make_vit_spatial_shard(
            (360, 720), 20, 4, sp1_rank, 3, sp2_rank, "sp1-sp2"
        )
        for sp2_rank in range(3)
        for sp1_rank in range(4)
    ]
    assert shards[0].h_splits == (100, 100, 80, 80)
    assert shards[0].w_splits == (240, 240, 240)
    assert shards[0].token_splits == (60, 60, 48, 48) * 3
    assert shards[-1].core_slices == (slice(280, 360), slice(480, 720))


def test_vit_policy_rejects_partial_patches():
    with pytest.raises(ValueError, match="not divisible"):
        make_vit_spatial_shard(
            (361, 720), 20, 2, 0, 1, 0, "sp1-sp2"
        )


def test_swin_policy_requires_complete_local_windows():
    shard = make_swin_spatial_shard(
        (128, 128), 4, (8, 8), 4, 0, 4, 0, "sp1-sp2"
    )
    assert shard.local_shape == (32, 32)
    with pytest.raises(ValueError, match="window height"):
        make_swin_spatial_shard(
            (96, 128), 4, (8, 8), 2, 0, 1, 0, "sp1-sp2"
        )


def test_spatial_objectives_match_single_rank_objectives():
    shard = make_vit_spatial_shard(
        (8, 6), 2, 1, 0, 1, 0, "sp1-sp2"
    )
    pred = torch.randn(2, 3, 8, 6)
    target = torch.randn_like(pred)
    torch.testing.assert_close(
        mse_loss(pred, target, shard.global_shape),
        torch.mean((pred - target) ** 2),
    )
    torch.testing.assert_close(
        rmse_spatial(pred, target, shard),
        rmse(pred, target),
    )


def _write_synthetic_data(tmp_path):
    fields = np.arange(5 * 2 * 8 * 6, dtype=np.float32).reshape(5, 2, 8, 6)
    for file_index in range(2):
        with h5py.File(
            tmp_path / f"trajectory_{file_index:04d}.h5", "w"
        ) as h5f:
            h5f.create_dataset("fields", data=fields)
    with h5py.File(tmp_path / "stats.h5", "w") as h5f:
        h5f.create_dataset("mean", data=np.zeros(2, dtype=np.float32))
        h5f.create_dataset("std", data=np.ones(2, dtype=np.float32))
    return fields


@pytest.mark.parametrize("read_mode", ["local", "all"])
def test_pytorch_loader_reads_the_resolved_core(tmp_path, read_mode):
    fields = _write_synthetic_data(tmp_path)
    shard = make_vit_spatial_shard(
        (8, 6), 2, 2, 1, 1, 0, "sp1-sp2"
    )
    params = Params(
        dt=1,
        n_in_channels=2,
        n_out_channels=2,
        img_size=(8, 6),
        spatial_read_mode=read_mode,
        limit_nsamples=None,
        limit_nsamples_val=None,
        valid_files=["9999"],
        stats_path=str(tmp_path / "stats.h5"),
    )
    dataset = SynDataset(params, str(tmp_path), train=True, spatial_shard=shard)
    inp, target = dataset[0]
    h_slice, w_slice = shard.core_slices
    torch.testing.assert_close(
        inp, torch.from_numpy(fields[0, :, h_slice, w_slice])
    )
    torch.testing.assert_close(
        target, torch.from_numpy(fields[1, :, h_slice, w_slice])
    )


def test_dali_external_source_local_and_full_reads_match(tmp_path):
    _write_synthetic_data(tmp_path)
    shard = make_vit_spatial_shard(
        (8, 6), 2, 2, 1, 1, 0, "sp1-sp2"
    )

    def make_source(read_mode):
        return SynES(
            str(tmp_path),
            False,
            1,
            1,
            (8, 6),
            2,
            2,
            1,
            0,
            None,
            spatial_shard=shard,
            spatial_read_mode=read_mode,
            enable_logging=False,
        )

    sample_info = SimpleNamespace(iteration=0, epoch_idx=0, idx_in_epoch=0)
    local = make_source("local")(sample_info)
    all_data = make_source("all")(sample_info)
    np.testing.assert_array_equal(local[0], all_data[0])
    np.testing.assert_array_equal(local[1], all_data[1])


def test_spatial_peers_read_different_cores_of_the_same_sample(tmp_path):
    fields = _write_synthetic_data(tmp_path)

    def make_source(spatial_rank):
        return SynES(
            str(tmp_path),
            False,
            1,
            1,
            (8, 6),
            2,
            2,
            1,
            0,
            None,
            spatial_shard=make_vit_spatial_shard(
                (8, 6),
                2,
                2,
                spatial_rank,
                1,
                0,
                "sp1-sp2",
            ),
            spatial_read_mode="local",
            enable_logging=False,
        )

    sample_info = SimpleNamespace(iteration=0, epoch_idx=0, idx_in_epoch=0)
    top = make_source(0)(sample_info)[0]
    bottom = make_source(1)(sample_info)[0]
    np.testing.assert_array_equal(
        np.concatenate((top, bottom), axis=-2), fields[0]
    )
