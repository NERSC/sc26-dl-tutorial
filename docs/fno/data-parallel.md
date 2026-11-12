# FNO: data parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Model parallel](model-parallel.md)

Each GPU still holds a full copy of the FNO and a different slice of the batch. The wiring is the same as the [ViT data-parallel page](../vit/data-parallel.md): [`submit_dp.sh`](../../submit_dp.sh) runs [`train.py`](../../train.py) under `torchrun`, one process per GPU, and `DistributedDataParallel` averages the gradients. This page starts at the larger model with local batch 1.

## A larger model at local batch 1

`short_opt_big` widens the small model to `embed_dim: 96` at `depth: 8`, 151.4M real parameters (the log reports 75.9M, counting each complex weight once). The ViT match at `depth: 4` and `downsample: 2` was 163 channels and 218.2M, and that forward does not fit at local batch 1. `downsample: 4` runs the blocks on a 1024×1024 latent grid. Global batch 4 on 4 GPUs, so the local batch is 1. Profile every rank.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_big \
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 ./submit_dp.sh --yaml_config ./config/FNO.yaml --config short_opt_big --run_num fno_big
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big/4GPU/fno_big/out.log`.

Profiles: `fno_big.0.nsys-rep` through `fno_big.3.nsys-rep`.

Submit this job.

`short_opt_big_dp16` keeps that local batch of 1 and uses 16 GPUs, so the global batch is 16. The learning rate follows the square-root rule, from `5e-4` to `1e-3`.

```bash
sbatch -N 2 ./submit_dp.sh --yaml_config ./config/FNO.yaml --config short_opt_big_dp16 --run_num fno_big16
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/fno_big16/out.log`.

Submit this job.

The next page splits the replica: [Model parallelism](model-parallel.md).
