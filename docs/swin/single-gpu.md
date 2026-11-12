# Swin: single GPU

[Tutorial](../../README.md) · [Data parallel](data-parallel.md) · [Model parallel](model-parallel.md)

Swin attends inside shifted windows instead of across every patch. The model size matches the [ViT single-GPU page](../vit/single-gpu.md): `embed_dim: 512`, `depth: 12`, `patch_size: 32`, and global batch 4. Swin's step is short enough that `short` trains on 256 samples and validates on 32, against 64 / 16 for the ViT, so each epoch is 64 steps instead of 16. `window_size: [16, 16]` divides the 128×128 patch grid. The knobs are the same three steps. [`config/swin.yaml`](../../config/swin.yaml) holds the blocks, and [`networks/swin.py`](../../networks/swin.py) is the model.

## Profiling

`train.py` wraps the host-to-device copy, the forward, and the optimizer step in the same NVTX ranges as the ViT. A profiled config still needs at least four epochs, because the capture is the fourth epoch.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short --run_num swin
```

Log: `$HOME/sc26-dl-tutorial/logs/short/1GPU/swin/out.log`. Profile: `swin_short.0.nsys-rep`.

The first three epochs look like this:

```text
2026-11-12 02:59:37,874 - root - INFO - Starting Training Loop...
2026-11-12 02:59:54,972 - root - INFO -  Memory usage after forward pass: 29.21038818359375 GB.
2026-11-12 03:02:45,816 - root - INFO - Time taken for epoch 1 is 165.053337 sec, avg 1.551014 samples/sec
2026-11-12 03:02:45,827 - root - INFO -   Avg train loss=0.214897
2026-11-12 03:02:45,842 - root - INFO -   Avg val loss=0.11220473796129227
2026-11-12 03:02:45,853 - root - INFO -   Total validation time: 11.5171639919281 sec
2026-11-12 03:02:49,238 - root - INFO -  Memory usage after forward pass: 31.39007568359375 GB.
2026-11-12 03:04:50,734 - root - INFO - Time taken for epoch 2 is 112.095470 sec, avg 2.283768 samples/sec
2026-11-12 03:04:50,745 - root - INFO -   Avg train loss=0.085521
2026-11-12 03:04:50,755 - root - INFO -   Avg val loss=0.10292744636535645
2026-11-12 03:04:50,765 - root - INFO -   Total validation time: 9.890985012054443 sec
2026-11-12 03:04:53,871 - root - INFO -  Memory usage after forward pass: 31.39007568359375 GB.
2026-11-12 03:06:49,758 - root - INFO - Time taken for epoch 3 is 106.194316 sec, avg 2.410675 samples/sec
2026-11-12 03:06:49,768 - root - INFO -   Avg train loss=0.074495
2026-11-12 03:06:49,779 - root - INFO -   Avg val loss=0.08669812977313995
2026-11-12 03:06:49,789 - root - INFO -   Total validation time: 9.893285036087036 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 31.390 GB peak.
- Throughput: 2.35 samples/sec.

The profiled epoch is mostly idle. Each step is a short burst of GPU work, and the gap before the next step is wider than the step itself.

![Swin baseline profile, wide idle gaps between steps](../../tutorial_images/swin_short_step.png)

## Workers, BF16, fused Adam, and `torch.compile`

`short_opt` turns on the same four changes as the ViT, except the worker count. The ViT uses 2 workers. Swin needs 4. BF16, fused Adam, and `torch.compile` are unchanged.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short_opt --run_num swin_opt
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt/1GPU/swin_opt/out.log`. Profile: `swin_short_opt.0.nsys-rep`.

The first epoch pays for `torch.compile`. The next two are the steady rate:

```text
2026-11-12 03:00:13,939 - root - INFO - Starting Training Loop...
2026-11-12 03:01:00,308 - root - INFO -  Memory usage after forward pass: 19.06787109375 GB.
2026-11-12 03:01:40,645 - root - INFO - Time taken for epoch 1 is 50.586947 sec, avg 5.060594 samples/sec
2026-11-12 03:01:40,656 - root - INFO -   Avg train loss=0.212684
2026-11-12 03:01:40,671 - root - INFO -   Avg val loss=0.11370135843753815
2026-11-12 03:01:40,682 - root - INFO -   Total validation time: 5.895118474960327 sec
2026-11-12 03:01:44,046 - root - INFO -  Memory usage after forward pass: 21.12451171875 GB.
2026-11-12 03:02:16,747 - root - INFO - Time taken for epoch 2 is 28.112366 sec, avg 9.106313 samples/sec
2026-11-12 03:02:16,757 - root - INFO -   Avg train loss=0.085068
2026-11-12 03:02:16,768 - root - INFO -   Avg val loss=0.1001790314912796
2026-11-12 03:02:16,778 - root - INFO -   Total validation time: 5.109506368637085 sec
2026-11-12 03:02:20,242 - root - INFO -  Memory usage after forward pass: 21.12451171875 GB.
2026-11-12 03:02:53,421 - root - INFO - Time taken for epoch 3 is 28.785261 sec, avg 8.893440 samples/sec
2026-11-12 03:02:53,431 - root - INFO -   Avg train loss=0.073881
2026-11-12 03:02:53,442 - root - INFO -   Avg val loss=0.08700230717658997
2026-11-12 03:02:53,452 - root - INFO -   Total validation time: 5.015381574630737 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 21.125 GB peak, down from 31.390 GB in the FP32 baseline.
- Throughput: 9.00 samples/sec, up from 2.35 samples/sec.

BF16 shortens the GPU work, and the gaps shrink with it. They are still there. Window attention is a much shorter step than the ViT forward, so even with 4 workers, instead of the ViT's 2, the host load between steps still shows up.

![Swin short_opt profile, shorter steps with gaps still between them](../../tutorial_images/swin_short_opt.png)

## NVIDIA DALI

`short_opt_dali` keeps those four settings and switches the loader to DALI.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short_opt_dali --run_num swin_dali
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_dali/1GPU/swin_dali/out.log`. Profile: `swin_short_opt_dali.0.nsys-rep`.

The first epoch still pays for `torch.compile`. The next two are the steady rate:

```text
2026-11-12 03:00:57,680 - root - INFO - Starting Training Loop...
2026-11-12 03:01:19,395 - root - INFO -  Memory usage after forward pass: 24.60101318359375 GB.
2026-11-12 03:01:41,749 - root - INFO - Time taken for epoch 1 is 25.304571 sec, avg 9.958675 samples/sec
2026-11-12 03:01:41,760 - root - INFO -   Avg train loss=0.218540
2026-11-12 03:01:41,765 - root - INFO -   Avg val loss=0.11386848241090775
2026-11-12 03:01:41,770 - root - INFO -   Total validation time: 3.258631944656372 sec
2026-11-12 03:01:42,287 - root - INFO -  Memory usage after forward pass: 26.66156005859375 GB.
2026-11-12 03:02:02,998 - root - INFO - Time taken for epoch 2 is 16.567833 sec, avg 15.451629 samples/sec
2026-11-12 03:02:03,009 - root - INFO -   Avg train loss=0.086518
2026-11-12 03:02:03,019 - root - INFO -   Avg val loss=0.10185879468917847
2026-11-12 03:02:03,024 - root - INFO -   Total validation time: 1.8626148700714111 sec
2026-11-12 03:02:03,538 - root - INFO -  Memory usage after forward pass: 26.66156005859375 GB.
2026-11-12 03:02:24,291 - root - INFO - Time taken for epoch 3 is 16.529154 sec, avg 15.487786 samples/sec
2026-11-12 03:02:24,301 - root - INFO -   Avg train loss=0.075379
2026-11-12 03:02:24,306 - root - INFO -   Avg val loss=0.0876319482922554
2026-11-12 03:02:24,312 - root - INFO -   Total validation time: 1.8594958782196045 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 26.662 GB peak, up from 21.125 GB with the PyTorch loader. Before the first step the device is already at 7.790 GB, against 1.224 GB for `short_opt`. That increase is DALI's copy of the prefetched batches.
- Throughput: 15.47 samples/sec, up from 9.00 samples/sec.

DALI closes most of those gaps. Its kernels run on a second stream while the step is in flight, and the steps sit against each other. That takes throughput from 9.00 to 15.47 samples/sec, on top of the 3.8× gain from `short_opt`.

![Swin DALI profile, steps packed together with DALI on a second stream](../../tutorial_images/swin_short_opt_dali.png)

The next page scales this step across GPUs: [Data parallelism](data-parallel.md).
