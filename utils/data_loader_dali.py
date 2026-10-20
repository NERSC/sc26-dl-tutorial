import h5py
import numpy as np
import torch
import torch.distributed as dist
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline

import utils.dali_es_helper as esh


def get_data_loader(
    params, data_path, distributed, train, spatial_shard=None
):
    dataloader = SynDaliDataLoader(
        params, data_path, train, spatial_shard=spatial_shard
    )

    if train:
        return dataloader, None, None
    else:
        return dataloader, None


class SynDaliDataLoader:
    def get_pipeline(self):
        pipeline = Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            device_id=self.device_index,
            py_num_workers=self.num_data_workers,
            py_start_method="spawn",
            seed=self.global_seed,
        )

        with pipeline:  # get input and target
            inp, tar = fn.external_source(
                source=esh.SynES(
                    self.data_path,
                    self.train,
                    self.batch_size,
                    self.dt,
                    self.img_size,
                    self.n_in_channels,
                    self.n_out_channels,
                    self.num_shards,
                    self.shard_id,
                    self.limit_nsamples,
                    spatial_shard=self.spatial_shard,
                    spatial_read_mode=self.spatial_read_mode,
                    enable_logging=False,
                    seed=self.global_seed,
                ),
                num_outputs=2,
                layout=["CHW", "CHW"],
                batch=False,
                parallel=True,
                prefetch_queue_depth=self.num_data_workers,
            )

            inp = inp.gpu()
            tar = tar.gpu()

            if self.normalize:
                inp = fn.normalize(
                    inp,
                    device="gpu",
                    axis_names="HW",
                    batch=False,
                    mean=self.in_bias,
                    stddev=self.in_scale,
                )

                tar = fn.normalize(
                    tar,
                    device="gpu",
                    axis_names="HW",
                    batch=False,
                    mean=self.out_bias,
                    stddev=self.out_scale,
                )

            pipeline.set_outputs(inp, tar)
        return pipeline

    def __init__(
        self, params, data_path, train, spatial_shard=None, seed=333
    ):
        self.global_seed = seed
        self.num_data_workers = params.num_data_workers
        self.device_index = torch.cuda.current_device()
        self.batch_size = int(params.local_batch_size)

        self.data_path = data_path
        self.train = train
        self.dt = params.dt
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.img_size = params.img_size
        self.spatial_shard = spatial_shard
        self.spatial_read_mode = params.get("spatial_read_mode", "local")
        self.limit_nsamples = (
            params.limit_nsamples if train else params.limit_nsamples_val
        )

        with h5py.File(params.stats_path, "r") as stats:
            means = stats["mean"][:].astype(np.float32)[:, None, None]
            stds = stats["std"][:].astype(np.float32)[:, None, None]
        self.normalize = True
        self.in_bias = means[: self.n_in_channels]
        self.in_scale = stds[: self.n_in_channels]
        self.out_bias = means[: self.n_out_channels]
        self.out_scale = stds[: self.n_out_channels]

        if dist.is_initialized():
            self.num_shards = params.data_num_shards
            self.shard_id = params.data_shard_id
        else:
            self.num_shards = 1
            self.shard_id = 0

        extsource = esh.SynES(
            self.data_path,
            self.train,
            self.batch_size,
            self.dt,
            self.img_size,
            self.n_in_channels,
            self.n_out_channels,
            self.num_shards,
            self.shard_id,
            self.limit_nsamples,
            spatial_shard=self.spatial_shard,
            spatial_read_mode=self.spatial_read_mode,
            seed=self.global_seed,
        )
        self.num_batches = extsource.num_steps_per_epoch
        del extsource

        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        self.iterator = DALIGenericIterator(
            [self.pipeline],
            ["inp", "tar"],
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
            prepare_first_batch=True,
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for token in self.iterator:
            inp = token[0]["inp"]
            tar = token[0]["tar"]

            yield inp, tar
