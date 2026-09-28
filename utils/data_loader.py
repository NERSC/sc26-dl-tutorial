import glob
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def worker_init(_worker_id):
    np.random.seed(torch.utils.data.get_worker_info().seed % (2**32 - 1))


def get_data_loader(params, data_path, distributed, train):
    dataset = SynDataset(params, data_path, train)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=train,
            num_replicas=params.data_num_shards,
            rank=params.data_shard_id,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.local_batch_size),
        num_workers=params.num_data_workers,
        shuffle=(sampler is None),
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class SynDataset(Dataset):
    def __init__(self, params, data_path, train):
        self.step = int(params.dt)
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.img_height, self.img_width = params.img_size
        self.limit_nsamples = (
            params.limit_nsamples if train else params.limit_nsamples_val
        )
        valid_files = {f"synthetic_{file_id}.h5" for file_id in params.valid_files}
        all_files = sorted(glob.glob(f"{data_path}/synthetic_*.h5"))
        if train:
            self.files_paths = [
                path for path in all_files if Path(path).name not in valid_files
            ]
        else:
            self.files_paths = [
                path for path in all_files if Path(path).name in valid_files
            ]
        if not self.files_paths:
            raise FileNotFoundError(f"No {'training' if train else 'validation'} files")

        with h5py.File(params.stats_path, "r") as stats:
            self.means = stats["mean"][:].astype(np.float32)[:, None, None]
            self.stds = stats["std"][:].astype(np.float32)[:, None, None]

        with h5py.File(self.files_paths[0], "r") as h5f:
            shape = h5f["fields"].shape
        self.samples_per_file = shape[0] - self.step
        if self.samples_per_file <= 0:
            raise ValueError("dt must be smaller than the number of steps per file")
        if self.img_height > shape[2] or self.img_width > shape[3]:
            raise ValueError("img_size is larger than the stored fields")

        self.n_samples_total = len(self.files_paths) * self.samples_per_file
        if self.limit_nsamples is not None:
            self.n_samples_total = min(self.n_samples_total, self.limit_nsamples)
        self.files = [None] * len(self.files_paths)
        logging.info(
            "Using %d files and %d samples for %s",
            len(self.files_paths),
            self.n_samples_total,
            "training" if train else "validation",
        )

    def _open_file(self, file_index):
        h5f = h5py.File(self.files_paths[file_index], "r")
        self.files[file_index] = h5f["fields"]

    def __len__(self):
        return self.n_samples_total

    def _normalize(self, img):
        img -= self.means
        img /= self.stds
        return torch.as_tensor(img)

    def __getitem__(self, global_idx):
        file_index, local_idx = divmod(global_idx, self.samples_per_file)
        if self.files[file_index] is None:
            self._open_file(file_index)
        fields = self.files[file_index]

        inp = fields[
            local_idx,
            : self.n_in_channels,
            : self.img_height,
            : self.img_width,
        ]
        tar = fields[
            local_idx + self.step,
            : self.n_out_channels,
            : self.img_height,
            : self.img_width,
        ]

        return self._normalize(inp), self._normalize(tar)
