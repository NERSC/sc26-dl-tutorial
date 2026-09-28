#!/usr/bin/env python3
"""Compute per-field mean and standard deviation with running statistics."""

from pathlib import Path

import h5py
import numpy as np


DATA_DIR = Path("/pscratch/sd/s/shas1693/data/sc26_tutorial_data")
ROWS_PER_BLOCK = 128


def print_file_contents(path):
    print(f"Contents of {path}:")
    with h5py.File(path, "r") as h5f:
        for name, item in h5f.items():
            print(f"  {name}: shape={item.shape}, dtype={item.dtype}")


def compute_stats(paths):
    count = 0
    mean = None
    m2 = None

    for file_number, path in enumerate(paths, start=1):
        print(f"Reading {path} ({file_number}/{len(paths)})")
        with h5py.File(path, "r") as h5f:
            fields = h5f["fields"]

            if mean is None:
                mean = np.zeros(fields.shape[1], dtype=np.float64)
                m2 = np.zeros(fields.shape[1], dtype=np.float64)

            for step in range(fields.shape[0]):
                for row in range(0, fields.shape[2], ROWS_PER_BLOCK):
                    block = fields[step, :, row : row + ROWS_PER_BLOCK, :]
                    block_count = block.shape[1] * block.shape[2]
                    block_mean = block.mean(axis=(1, 2), dtype=np.float64)
                    block_m2 = np.sum(
                        (block - block_mean[:, None, None]) ** 2,
                        axis=(1, 2),
                        dtype=np.float64,
                    )

                    new_count = count + block_count
                    delta = block_mean - mean
                    mean += delta * block_count / new_count
                    m2 += block_m2 + delta**2 * count * block_count / new_count
                    count = new_count

    return mean, np.sqrt(m2 / count), count


if __name__ == "__main__":
    files = sorted(DATA_DIR.glob("synthetic_*.h5"))
    if not files:
        raise FileNotFoundError(f"No synthetic HDF5 files found in {DATA_DIR}")

    print_file_contents(files[0])
    mean, std, count = compute_stats(files)

    output_path = DATA_DIR / "stats.h5"
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("mean", data=mean)
        h5f.create_dataset("std", data=std)
        h5f.attrs["samples_per_field"] = count

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print(f"Saved {output_path}")
