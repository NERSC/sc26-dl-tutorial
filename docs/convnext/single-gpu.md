# ConvNeXt: single GPU

[Tutorial](../../README.md) · [Data parallel](data-parallel.md) · [Spatial parallel](model-parallel.md) · [Runbook](../RUNBOOK.md#convnext)

ConvNeXt mixes nearby feature cells with a depthwise convolution, then mixes
channels with an MLP. The [serial model](../../networks/convnext.py) keeps the
same feature-grid resolution through all blocks. A patch stem turns each
4096×4096, two-channel field into a 128×128 grid; the head reconstructs the
two-channel field. This implementation uses zero padding at the field boundary.

The three stages in [ConvNeXt.yaml](../../config/ConvNeXt.yaml) use patch 32,
width 512, depth 12, kernel 7, batch 4, and Adam with learning rate `5e-4`.
They follow the [Swin optimization ladder](../swin/single-gpu.md).

| Config | Precision | Loader / workers | Fused Adam | Compile | Iterations |
|---|---|---|---|---|---|
| `short` | FP32 | PyTorch / 0 | No | No | 320 |
| `short_opt` | BF16 autocast | PyTorch / 4 | Yes | Yes | 320 |
| `short_opt_dali` | BF16 autocast | DALI / 4 | Yes | Yes | 320 |

Each run uses 256 training and 32 validation samples. `num_iters: 320`
schedules five epochs.

## Profiling

`train.py` marks data loading, device transfers, the forward pass, and optimizer
updates. The ConvNeXt block adds `depthwise convolution` and `mlp` ranges, as in
ViT. The spatial model uses the same block ranges.

The first epoch includes startup and compilation. Epoch 4 is the profiled
epoch. Throughput measurements leave out epoch 1 and the profiled epoch.

Compare the gaps between steps and the kernel durations across the three
stages. DALI can overlap loading with compute; its prefetching also uses GPU
memory. The [runbook](../RUNBOOK.md#convnext) has the submission commands and
output paths.

Continue with [data parallelism](data-parallel.md).
