# ConvNeXt: spatial parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md)

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

## A larger model at local batch 1

`short_mp` uses patch 16, width 1536, depth 16, and kernel 7: 304.996 million
parameters. BF16, fused Adam, DALI with two workers, and eager execution are
enabled. The feature grid is 256×256. Global batch stays 4, with 64 training
and 16 validation samples; 80 iterations schedule five epochs.

Both runs use [train_mp.py](../../train_mp.py) and four data-parallel replicas,
so each replica processes one sample per step. Spatial splitting distributes
that sample's feature map across GPUs; it does not divide the batch.

| Run | GPUs | DP replicas | Spatial split | Local / global batch |
|---|---|---|---|---|
| No split | 4 | 4 | `sp1=1, sp2=1` | 1 / 4 |
| Spatial parallel | 16 | 4 | `sp1=4, sp2=1` | 1 / 4 |

## No split

Start with one complete model and one sample per GPU. This is the
batch-1 memory comparison for the large model.

```bash
ENABLE_PROFILING=0 \
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 ./submit_mp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_mp --run_num convnext_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/1MP/convnext_oom/out.log`.

Submit this job. Check whether the full model fits at local batch 1 and
inspect the job output if it runs out of memory.

## Four-way spatial split on sixteen GPUs

Keep the same model and global batch, and split each replica across four GPUs.
Each rank holds a 64×256 feature tile. Two eight-GPU nodes hold four replicas,
with local batch 1 and global batch 4. The learning rate remains `5e-4`.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_mp \
sbatch -N 2 ./submit_mp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_mp --run_num convnext_mp \
    --tp=1 --sp1=4 --sp2=1
```

The default rank order, `tp-sp1-sp2-dp`, keeps each spatial group within a node.
The trainer infers the number of data-parallel replicas from the GPU count
and spatial mesh.

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/4MP/convnext_mp/out.log`.

Profiles: `convnext_mp.0.nsys-rep` through `convnext_mp.15.nsys-rep`.

Submit this job. Compare memory use with the unsharded run, check that
training and validation complete, and inspect the profile for halo exchanges
during the forward pass and gradient synchronization during backward.
