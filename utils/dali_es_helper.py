import glob
import logging

import h5py
import numpy as np


class SynES:
    def __init__(
        self,
        data_path,
        train,
        batch_size,
        dt,
        img_size,
        n_in_channels,
        n_out_channels,
        num_shards,
        shard_id,
        limit_nsamples,
        spatial_shard=None,
        spatial_read_mode="local",
        enable_logging=True,
        seed=333,
    ):
        self.batch_size = batch_size
        self.data_path = data_path
        self.img_size = img_size
        self.train = train
        self.step = int(dt)
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.rng = np.random.default_rng(seed)
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.limit_nsamples = limit_nsamples
        self.spatial_shard = spatial_shard
        self.spatial_read_mode = spatial_read_mode
        if self.spatial_read_mode not in {"local", "all"}:
            raise ValueError(
                "spatial_read_mode must be 'local' or 'all'"
            )

        self._get_files_stats(enable_logging)
        self.shuffle = train

    def _get_files_stats(self, enable_logging):
        all_files = sorted(
            glob.glob(f"{self.data_path}/trajectory_*.h5")
        )
        split_index = int(0.9 * len(all_files))
        self.files_paths = (
            all_files[:split_index]
            if self.train
            else all_files[split_index:]
        )
        if not self.files_paths:
            raise FileNotFoundError(
                f"No {'training' if self.train else 'validation'} files"
            )

        with h5py.File(self.files_paths[0], "r") as h5f:
            shape = h5f["fields"].shape
            self.img_shape_x = self.img_size[0]
            self.img_shape_y = self.img_size[1]
        self.samples_per_file = shape[0] - self.step
        if self.samples_per_file <= 0:
            raise ValueError("dt must be smaller than the number of steps per file")
        if self.img_shape_x > shape[2] or self.img_shape_y > shape[3]:
            raise ValueError("img_size is larger than the stored fields")

        self.n_samples_total = len(self.files_paths) * self.samples_per_file
        if self.limit_nsamples is not None:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.files = [None] * len(self.files_paths)
        self.dsets = [None] * len(self.files_paths)
        if enable_logging:
            logging.info(
                "Using %d files and %d samples for %s",
                len(self.files_paths),
                self.n_samples_total,
                "training" if self.train else "validation",
            )

        self.num_steps_per_epoch = self.n_samples_shard // self.batch_size
        self.last_epoch = None
        self.index_permutation = None
        self.inp_buf = None
        self.tar_buf = None
        self.full_buf = None

    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        for f in self.files:
            if f is not None:
                f.close()

    def __call__(self, sample_info):
        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        if sample_info.epoch_idx != self.last_epoch:
            self.last_epoch = sample_info.epoch_idx
            if self.shuffle:
                self.index_permutation = self.rng.permutation(self.n_samples_total)
            else:
                self.index_permutation = np.arange(self.n_samples_total)
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]

        sample_idx = self.index_permutation[sample_info.idx_in_epoch]
        file_index, local_idx = divmod(sample_idx, self.samples_per_file)
        if self.files[file_index] is None:
            self.files[file_index] = h5py.File(self.files_paths[file_index], "r")
            self.dsets[file_index] = self.files[file_index]["fields"]

        if self.spatial_shard is None:
            local_shape = (self.img_shape_x, self.img_shape_y)
            h_slice = slice(0, self.img_shape_x)
            w_slice = slice(0, self.img_shape_y)
        else:
            local_shape = self.spatial_shard.local_shape
            h_slice, w_slice = self.spatial_shard.core_slices

        if self.inp_buf is None:
            self.inp_buf = np.empty(
                (self.n_in_channels, *local_shape),
                dtype=np.float32,
            )
            self.tar_buf = np.empty(
                (self.n_out_channels, *local_shape),
                dtype=np.float32,
            )

        fields = self.dsets[file_index]
        if self.spatial_read_mode == "local":
            fields.read_direct(
                self.inp_buf,
                source_sel=np.s_[
                    local_idx,
                    : self.n_in_channels,
                    h_slice,
                    w_slice,
                ],
            )
            fields.read_direct(
                self.tar_buf,
                source_sel=np.s_[
                    local_idx + self.step,
                    : self.n_out_channels,
                    h_slice,
                    w_slice,
                ],
            )
        else:
            if self.full_buf is None:
                self.full_buf = np.empty(
                    (
                        max(self.n_in_channels, self.n_out_channels),
                        self.img_shape_x,
                        self.img_shape_y,
                    ),
                    dtype=np.float32,
                )
            fields.read_direct(
                self.full_buf[: self.n_in_channels],
                source_sel=np.s_[
                    local_idx,
                    : self.n_in_channels,
                    : self.img_shape_x,
                    : self.img_shape_y,
                ],
            )
            self.inp_buf[...] = self.full_buf[
                : self.n_in_channels, h_slice, w_slice
            ]
            fields.read_direct(
                self.full_buf[: self.n_out_channels],
                source_sel=np.s_[
                    local_idx + self.step,
                    : self.n_out_channels,
                    : self.img_shape_x,
                    : self.img_shape_y,
                ],
            )
            self.tar_buf[...] = self.full_buf[
                : self.n_out_channels, h_slice, w_slice
            ]

        # DALI consumes samples asynchronously, so each result needs independent
        # storage that will not be overwritten by the next call.
        return self.inp_buf.copy(), self.tar_buf.copy()
