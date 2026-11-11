# ConvNeXt: data parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Spatial parallel](model-parallel.md) · [Runbook](../RUNBOOK.md#convnext)

Each GPU holds the full patch-32, width-512, depth-12, kernel-7 ConvNeXt and
receives different samples. [train.py](../../train.py) uses
DistributedDataParallel to average gradients. Keep the optimized single-GPU
settings: BF16 autocast, fused Adam, compilation, and DALI with four workers.

| Config | Nodes / GPUs | Local / global train batch | Learning rate | Train / validation batches per epoch | Updates for 5 epochs |
|---|---|---|---|---|---|
| `short_opt_dali` | 1 / 1 | 4 / 4 | `5e-4` | 64 / 8 | 320 |
| `short_opt_dali_dp4` | 1 / 4 | 4 / 16 | `1e-3` | 16 / 2 | 80 |
| `short_opt_dali_dp16` | 4 / 16 | 4 / 64 | `2e-3` | 4 / 2 | 20 |

The learning rate is `5e-4 * sqrt(global_batch / 4)`: factors 2 and 4 at
global batches 16 and 64. Each run covers five scheduled epochs, with a cosine
learning-rate schedule over the listed number of updates.

The 1- and 4-GPU runs validate on 32 samples. The 16-GPU config uses 128 so
each rank has two validation batches of 4.

## Scaling

For mean global throughput `T`, speedup is `T_N / T_1` and efficiency is
`T_N / (N * T_1)`. The trainer reports global samples/sec. Throughput
measurements leave out epoch 1 and the profiled epoch.

Profile every rank. Epoch 4 is the profiled epoch. In Nsight, compare gradient
reductions, their overlap with backward computation, and input stalls. The
[runbook](../RUNBOOK.md#convnext) has the submission commands and output paths.
