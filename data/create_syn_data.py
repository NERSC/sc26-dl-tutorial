#!/usr/bin/env python3
"""Create a synthetic four-field spatiotemporal dataset."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter


# Dataset settings
OUTPUT_DIR = Path("/pscratch/sd/s/shas1693/data/sc26_tutorial_data")
HEIGHT = 4096
WIDTH = 4096
NUM_STEPS = 256
STEPS_PER_FILE = 16
NUM_FIELDS = 4
DT = 0.05
ROWS_PER_BLOCK = 128

# GIF settings: 256 frames at 16 FPS gives a 16-second animation.
GIF_SIZE = 256
GIF_FPS = 16

# (x frequency, y frequency, time frequency, amplitude, phase)
MODES = [
    [(1, 2, 0.55, 1.0, 0.0), (4, -3, 1.10, 0.45, 0.7), (9, 5, -0.75, 0.22, 1.3)],
    [(2, -1, 0.40, 0.9, 0.4), (5, 4, -1.25, 0.50, 1.1), (11, -7, 0.90, 0.18, 2.0)],
    [(1, 6, 1.00, 0.8, 1.2), (7, 2, 0.65, 0.55, 0.1), (13, -9, -1.60, 0.20, 2.4)],
    [(3, 3, -0.50, 0.75, 0.2), (8, -5, 0.95, 0.45, 1.5), (15, 4, 1.80, 0.20, 2.7)],
]


def make_field(x, y, time, field_index):
    """Sum traveling Fourier modes with known time dependence."""
    field = np.zeros((len(y), len(x)), dtype=np.float32)
    for kx, ky, omega, amplitude, phase in MODES[field_index]:
        field += amplitude * np.sin(kx * x[None, :] + ky * y[:, None] - omega * time + phase)
    return field


def write_hdf5_files():
    x = np.linspace(0, 2 * np.pi, WIDTH, endpoint=False, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi, HEIGHT, endpoint=False, dtype=np.float32)
    num_files = NUM_STEPS // STEPS_PER_FILE

    for file_index in range(num_files):
        path = OUTPUT_DIR / f"synthetic_{file_index:04d}.h5"
        print(f"Writing {path} ({file_index + 1}/{num_files})")

        with h5py.File(path, "w") as h5f:
            fields = h5f.create_dataset(
                "fields",
                shape=(STEPS_PER_FILE, NUM_FIELDS, HEIGHT, WIDTH),
                dtype=np.float32,
                chunks=(1, 1, ROWS_PER_BLOCK, WIDTH),
            )

            first_step = file_index * STEPS_PER_FILE
            steps = np.arange(first_step, first_step + STEPS_PER_FILE)
            h5f.create_dataset("time", data=steps * DT)
            h5f.attrs["dt"] = DT
            h5f.attrs["domain"] = "[0, 2*pi) x [0, 2*pi), periodic"

            for local_step, step in enumerate(steps):
                for row in range(0, HEIGHT, ROWS_PER_BLOCK):
                    y_block = y[row : row + ROWS_PER_BLOCK]
                    for field_index in range(NUM_FIELDS):
                        fields[local_step, field_index, row : row + ROWS_PER_BLOCK] = (
                            make_field(x, y_block, step * DT, field_index)
                        )
                print(f"  Step {step + 1}/{NUM_STEPS}", flush=True)


def write_gif():
    x = np.linspace(0, 2 * np.pi, GIF_SIZE, endpoint=False, dtype=np.float32)
    y = np.linspace(0, 2 * np.pi, GIF_SIZE, endpoint=False, dtype=np.float32)
    limits = [sum(mode[3] for mode in field_modes) for field_modes in MODES]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    images = []
    for index, axis in enumerate(axes.flat):
        image = axis.imshow(
            make_field(x, y, 0, index),
            origin="lower",
            cmap="turbo",
            vmin=-limits[index],
            vmax=limits[index],
        )
        axis.set_title(f"Field {index}")
        axis.axis("off")
        images.append(image)

    title = fig.suptitle("")
    gif_path = Path(__file__).resolve().parent / "synthetic_evolution.gif"
    print(f"Writing {gif_path} at {GIF_FPS} FPS")

    writer = PillowWriter(fps=GIF_FPS)
    with writer.saving(fig, gif_path, dpi=100):
        for step in range(NUM_STEPS):
            time = step * DT
            for field_index, image in enumerate(images):
                image.set_data(make_field(x, y, time, field_index))
            title.set_text(f"Step {step}, t = {time:.2f}")
            writer.grab_frame()

    plt.close(fig)


if __name__ == "__main__":
    raw_bytes = NUM_STEPS * NUM_FIELDS * HEIGHT * WIDTH * 4
    print(f"Raw FP32 size: {raw_bytes / 1024**3:.1f} GiB")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_hdf5_files()
    write_gif()
