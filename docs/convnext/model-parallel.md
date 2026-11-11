# ConvNeXt: spatial parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md) · [Runbook](../RUNBOOK.md#convnext)

[convnext_mp.py](../../networks/convnext_mp.py) splits feature maps across
spatial tiles. Weights, gradients, and optimizer states remain replicated.
`tp=1` is required: this ConvNeXt does not implement tensor parallelism. Spatial
splitting reduces activation storage and the per-rank input/output tile size,
but does not divide parameter storage.

A 7×7 depthwise convolution needs a halo of three feature cells. Each block
exchanges width stripes (`sp2`), then height stripes (`sp1`), including the
corners. Global edges keep the serial model's zero padding. Backward returns
and sums the halo gradients. The channel MLP and normalization stay local;
weight gradients are summed across spatial ranks and averaged across replicas.

## A larger model

`short_mp` uses patch 16, width 768, depth 12, kernel 7, BF16, fused Adam,
DALI with two workers, and eager execution. The feature grid is 256×256.
The batch is 4, with 256 training and 32 validation samples. `num_iters: 320`
schedules five epochs.

Both the unsharded and spatial runs use [train_mp.py](../../train_mp.py).
On one GPU, `tp=1, sp1=1, sp2=1` keeps the complete field on the device.
On four GPUs, `tp=1, sp1=4, sp2=1` gives each rank a 64×256 feature tile.
Both runs have one DP replica and the same batch of four samples.

## Four replicas across four nodes

`short_mp_dp4` keeps the same model and `tp=1, sp1=4, sp2=1`. Sixteen GPUs
leave `dp=4`: four replicas, each split across one node. Local batch stays 4
and global batch becomes 16. The learning rate is `1e-3`, from
`5e-4 * sqrt(16 / 4)`. There are 16 training and 2 validation batches per epoch,
so `num_iters: 80` schedules five epochs.

The default rank order, `tp-sp1-sp2-dp`, keeps each spatial group within one
node. The trainer infers DP size from world size and the spatial mesh.

Throughput measurements leave out epoch 1 and the profiled epoch.
Replica speedup is `T_4nodes / T_1node`; efficiency is that ratio divided by 4.
In the profiles, look for halo exchanges around the depthwise convolutions,
shared-weight and DP reductions, and validation gathers.

The [runbook](../RUNBOOK.md#convnext) has the submission commands and output paths.
