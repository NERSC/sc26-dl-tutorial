#!/usr/bin/env python3
"""Compute trajectory statistics from the training-file split."""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATA_DIR = Path("/pscratch/sd/s/shas1693/data/sc26_tutorial_data")
DEFAULT_PATTERN = "trajectory_*.h5"
DEFAULT_TRAIN_FRACTION = 0.9
ROWS_PER_BLOCK = 128


def compute_stats(paths):
    """Compute population mean/std per channel without loading full files."""
    count = 0
    mean = None
    m2 = None

    for file_number, path in enumerate(paths, start=1):
        print(f"Reading {path} ({file_number}/{len(paths)})", flush=True)
        with h5py.File(path, "r") as h5f:
            fields = h5f["fields"]

            if mean is None:
                mean = np.zeros(fields.shape[1], dtype=np.float64)
                m2 = np.zeros(fields.shape[1], dtype=np.float64)
            elif fields.shape[1] != mean.size:
                raise ValueError(
                    f"{path} has {fields.shape[1]} channels; expected {mean.size}"
                )

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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument(
        "--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1")

    files = sorted(args.data_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} found in {args.data_dir}"
        )

    split_index = int(len(files) * args.train_fraction)
    if split_index == 0 or split_index == len(files):
        raise ValueError("Split must contain both training and validation files")
    train_files = files[:split_index]
    val_files = files[split_index:]
    print(
        f"Using {len(train_files)} training files ({train_files[0].name} through "
        f"{train_files[-1].name}); reserving {len(val_files)} validation files "
        f"({val_files[0].name} through {val_files[-1].name})",
        flush=True,
    )

    mean, std, count = compute_stats(train_files)
    output_path = args.output or args.data_dir / "stats_trajectory.h5"
    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("mean", data=mean)
        h5f.create_dataset("std", data=std)
        h5f.attrs["samples_per_field"] = count
        h5f.attrs["num_train_files"] = len(train_files)
        h5f.attrs["train_fraction"] = args.train_fraction
        h5f.attrs["file_pattern"] = args.pattern

    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
