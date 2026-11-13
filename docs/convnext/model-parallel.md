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

## Spatial parallel data loading (optional)

A local tile read can help when the field is large. Each rank reads an HDF5
hyperslab of its own tile, input and target, and no collective follows. One
4096×4096 sample is 256 MB; the four-way height split brings that read down to
64 MB per rank. On a small image the saving is too small to pay for the
hyperslab. These runs leave `sp2=1`, so each local read is a block of whole
rows. With `sp2` greater than 1, that same local read takes a width tile: a
strided hyperslab in the file, and a slow read.

ConvNeXt turns this on. `base` in [ConvNeXt.yaml](../../config/ConvNeXt.yaml)
sets `spatial_read_mode: local`, and `short_mp` inherits it. ViT and Swin leave
the key at `all`, which reads the full field and keeps this rank's tile. The
FNO config is also `local`. Toggle the key to `all` on the sixteen-GPU job and
compare the data-load gap in the profile, to see where the tile read helps.

