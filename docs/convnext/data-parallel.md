# ConvNeXt: data parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Spatial parallel](model-parallel.md)

Each GPU holds a full copy of the model and receives different samples.
[submit_dp.sh](../../submit_dp.sh) launches [train.py](../../train.py) with
one process per GPU, and DistributedDataParallel averages the gradients.
Keep BF16, fused Adam, compilation, and DALI with two workers enabled.

## Scaling the batch

`short_opt_dali_dp` keeps patch 32, width 512, depth 12, and kernel 7.
Global batch 16 on four GPUs gives local batch 4, matching the single-GPU
DALI run. The learning rate stays `5e-4`. With 64 training and 16 validation
samples, 80 iterations schedule twenty epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_dp4 \
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 ./submit_dp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_opt_dali_dp --run_num convnext_dp4
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_dali_dp/4GPU/convnext_dp4/out.log`.

Profiles: `convnext_dp4.0.nsys-rep` through `convnext_dp4.3.nsys-rep`.

Submit this job. Compare global throughput with the single-GPU DALI run,
and look for gradient synchronization during backward in the profile.

## A larger model at local batch 1

`short_opt_big` uses patch 16, width 1024, depth 12, and kernel 7. The feature
grid grows to 256×256. Global batch 4 on four GPUs gives local batch 1.
The run uses 64 training and 16 validation samples, with 80 iterations
scheduling five epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_big \
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 ./submit_dp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_opt_big --run_num convnext_big
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big/4GPU/convnext_big/out.log`.

Profiles: `convnext_big.0.nsys-rep` through `convnext_big.3.nsys-rep`.

Submit this job. Check memory use and GPU activity with the larger model.

`short_opt_big_dp16` keeps the same model and local batch 1 on sixteen GPUs.
Global batch becomes 16, and the learning rate is `1e-3`, from
`5e-4 * sqrt(16 / 4)`. The sample counts stay the same; 80 iterations schedule
twenty epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_big16 \
sbatch -N 2 ./submit_dp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_opt_big_dp16 --run_num convnext_big16
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/convnext_big16/out.log`.

Profiles: `convnext_big16.0.nsys-rep` through `convnext_big16.15.nsys-rep`.

Submit this job. Compare its throughput with the four-GPU run to measure
scaling at the same local batch. Use epochs 2, 3, and 5 for both comparisons.

Continue with [spatial parallelism](model-parallel.md).
