# Swin: data parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Model parallel](model-parallel.md)

Each GPU still holds a full copy of the Swin and a different slice of the batch. The wiring is the same as the [ViT data-parallel page](../vit/data-parallel.md): [`submit_dp.sh`](../../submit_dp.sh) runs [`train.py`](../../train.py) with four tasks per node, and `DistributedDataParallel` averages the gradients. Window attention does not change that. The interactive commands are steps 4–6 under Swin in [RUNBOOK_run.md](../RUNBOOK_run.md).

## Scaling the batch

`short_opt_dali_dp` keeps the local batch at 4. Four GPUs make the global batch 16. The learning rate stays `5e-4`.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_dp \
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_dali_dp --run_num swin_dp
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali_dp/4GPU/swin_dp/out.log`. Profiles: `swin_dp.0.nsys-rep` through `swin_dp.3.nsys-rep`.

> **TODO:** first three epochs, per-GPU memory, and samples/sec. The gradient all-reduce should sit on its own stream during the backward, as it does for the ViT.

## A larger model at local batch 1

`short_opt_big` uses the same size-up as the ViT: `patch_size: 16`, `embed_dim: 1024`, `depth: 12`. The patch grid is 256×256, and `window_size` 16 divides it. Global batch 4 on 4 GPUs, so the local batch is 1. No profile.

```bash
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big --run_num swin_big
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big/4GPU/swin_big/out.log`.

> **TODO:** throughput at the last epoch.

`short_opt_big_dp16` keeps that local batch of 1 and uses 16 GPUs, so the global batch is 16. The learning rate follows the square-root rule, from `5e-4` to `1e-3`.

```bash
sbatch -N 4 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big_dp16 --run_num swin_big16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/swin_big16/out.log`.

> **TODO:** throughput of the 16-GPU run.

The next page splits the replica: [Model parallelism](model-parallel.md).
