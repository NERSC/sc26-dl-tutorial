"""Training helpers shared by the serial and model-parallel entry points."""

import torch


def profiled_batches(loader, split):
    """Yield batches from ``loader``, marking iterator and fetch ranges for NVTX."""
    with torch.cuda.nvtx.range(f"data/{split}/iterator"):
        iterator = iter(loader)
    while True:
        with torch.cuda.nvtx.range(f"data/{split}/next_batch"):
            try:
                batch = next(iterator)
            except StopIteration:
                return
        yield batch


def get_data_loader_distributed(
    params, location, distributed, train, spatial_shard=None
):
    """Build the PyTorch or DALI loader selected by ``data_loader_config``."""
    if params.data_loader_config.startswith("dali"):
        from .data_loader_dali import get_data_loader
    else:
        from .data_loader import get_data_loader
    return get_data_loader(
        params, location, distributed, train, spatial_shard=spatial_shard
    )
