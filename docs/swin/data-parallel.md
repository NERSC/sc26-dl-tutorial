# Swin: data parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Model parallel](model-parallel.md)

Each GPU still holds a full copy of the Swin and a different slice of the batch. The wiring is the same as the [ViT data-parallel page](../vit/data-parallel.md): [`submit_dp.sh`](../../submit_dp.sh) runs [`train.py`](../../train.py) under `torchrun`, one process per GPU, and `DistributedDataParallel` averages the gradients. Window attention is a short step, so this page starts at the larger model with local batch 1.

## A larger model at local batch 1

`short_opt_big` uses the same size-up as the ViT: `patch_size: 16`, `embed_dim: 1024`, `depth: 12`. The patch grid is 256×256, and `window_size` 16 divides it. Global batch 4 on 4 GPUs, so the local batch is 1. Profile every rank.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt_dali_big \
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big --run_num swin_big_profile
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big/4GPU/swin_big_profile/out.log`.

Profiles: `swin_short_opt_dali_big.0.nsys-rep` through `swin_short_opt_dali_big.3.nsys-rep`.

The first epoch pays for `torch.compile`. Epoch 4 is the profiled epoch.

```text
2026-11-12 03:02:13,016 - root - INFO - Starting Training Loop...
2026-11-12 03:02:42,175 - root - INFO -  Memory usage after forward pass: 36.4117431640625 GB.
2026-11-12 03:03:41,369 - root - INFO - Time taken for epoch 1 is 65.220147 sec, avg 3.863837 samples/sec
2026-11-12 03:03:41,379 - root - INFO -   Avg train loss=0.279806
2026-11-12 03:03:41,385 - root - INFO -   Avg val loss=0.1547510176897049
2026-11-12 03:03:41,390 - root - INFO -   Total validation time: 3.2699644565582275 sec
2026-11-12 03:03:42,281 - root - INFO -  Memory usage after forward pass: 36.9117431640625 GB.
2026-11-12 03:04:25,209 - root - INFO - Time taken for epoch 2 is 39.418287 sec, avg 6.494448 samples/sec
2026-11-12 03:04:25,219 - root - INFO -   Avg train loss=0.116139
2026-11-12 03:04:25,230 - root - INFO -   Avg val loss=0.1373618096113205
2026-11-12 03:04:25,240 - root - INFO -   Total validation time: 1.6988492012023926 sec
2026-11-12 03:04:26,088 - root - INFO -  Memory usage after forward pass: 37.1617431640625 GB.
2026-11-12 03:05:09,120 - root - INFO - Time taken for epoch 3 is 39.357143 sec, avg 6.504537 samples/sec
2026-11-12 03:05:09,130 - root - INFO -   Avg train loss=0.100376
2026-11-12 03:05:09,141 - root - INFO -   Avg val loss=0.11545325815677643
2026-11-12 03:05:09,146 - root - INFO -   Total validation time: 1.8019866943359375 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 37.162 GB peak after the forward pass.
- Throughput: 6.50 samples/sec across 4 GPUs, with local batch 1.

The gradient all-reduces are NCCL kernels on their own stream, issued during the backward. The larger Swin makes the step long enough that DALI hides the read, and the timeline shows no idle gap between steps.

![Swin data-parallel profile, NCCL gradient all-reduce during the backward](../../tutorial_images/swin_big.png)

`short_opt_big_dp16` keeps that local batch of 1 and uses 16 GPUs, so the global batch is 16. The learning rate follows the square-root rule, from `5e-4` to `1e-3`.

```bash
sbatch -N 2 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big_dp16 --run_num swin_big16
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/swin_big16/out.log`.

After the first epoch the run holds 24.2 samples/sec, up from 6.50 on 4 GPUs. The forward peaks at 36.218 GB.

The next page splits the replica: [Model parallelism](model-parallel.md).
