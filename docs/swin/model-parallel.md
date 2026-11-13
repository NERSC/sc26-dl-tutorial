# Swin: model parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md)

The mesh is the same one documented on the [ViT model-parallel page](../vit/model-parallel.md): `dp × tp × sp1 × sp2`, with column-parallel and row-parallel linears over `tp`. Swin keeps `sp1` and `sp2` as separate spatial axes. A block of windows lives on each spatial rank. Heads are the tensor-parallel axis. The shifted-window blocks exchange the cells that cross a rank boundary through `roll_forward_reverseroll_backward` in [`networks/swin_mp.py`](../../networks/swin_mp.py): the forward rolls those cells across, and the backward rolls the gradient back. `window_size` has to divide the patch grid on each spatial rank. 

## No split

`short_mp` is the same size-up that ran out of memory for the ViT: `patch_size: 16`, `embed_dim: 1536`, `depth: 16`, local batch 1. The patch grid is 256×256, which 16 divides. `--tp=1 --sp1=1 --sp2=1` leaves the whole replica on every GPU. Window attention stores less activation memory than full attention. It is not enough: the forward still runs out of memory, in the local `torch.roll` inside `_distributed_roll`. With one spatial rank that roll never leaves the GPU. Profiling stays off.

```bash
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 \
    ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/1MP/swin_oom/out.log`.

The model is built before the step fails. The log reports a local parameter count of 555.545 M.

```text
File "distributed/helpers.py", line 277, in _distributed_roll
    return torch.roll(input_, shifts=shift, dims=dim)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 384.00 MiB. GPU 2 has a total capacity of 39.49 GiB of which 228.00 MiB is free. Including non-PyTorch memory, this process has 39.26 GiB memory in use. Of the allocated memory 34.01 GiB is allocated by PyTorch, and 176.98 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Spatial split

`sp1=4` shards one spatial axis. The local grid is 64×256, and window 16 divides both sides. The shift on odd blocks is half a window, 8, which is inside that local grid. Two nodes leave `dp=4`, so the global batch of 4 keeps the local batch at 1. A group of size 1 on `sp2` rolls locally.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_mp \
sbatch -N 2 ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_mp \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/4MP/swin_mp/out.log`. Profiles: `swin_mp.0.nsys-rep` through `swin_mp.15.nsys-rep`.

`sp1=4` brings the local parameter count to 480.048 M. The first three epochs stay under 34 GB. Epoch 4 is the profiled epoch.

```text
2026-11-12 03:19:33,633 - root - INFO - Starting Training Loop...
2026-11-12 03:19:38,155 - root - INFO -  Memory usage after forward pass: 31.08612060546875 GB.
2026-11-12 03:20:12,485 - root - INFO - Time taken for epoch 1 is 31.518902 sec, avg 7.995202 samples/sec
2026-11-12 03:20:12,495 - root - INFO -   Avg train loss=0.374362
2026-11-12 03:20:12,500 - root - INFO -   Avg val loss=0.3081926703453064
2026-11-12 03:20:12,500 - root - INFO -   Total validation time: 1.1185808181762695 sec
2026-11-12 03:20:13,336 - root - INFO -  Memory usage after forward pass: 33.27947998046875 GB.
2026-11-12 03:20:47,744 - root - INFO - Time taken for epoch 2 is 31.457812 sec, avg 8.137883 samples/sec
2026-11-12 03:20:47,755 - root - INFO -   Avg train loss=0.229070
2026-11-12 03:20:47,765 - root - INFO -   Avg val loss=0.14864471554756165
2026-11-12 03:20:47,770 - root - INFO -   Total validation time: 1.1535515785217285 sec
2026-11-12 03:20:48,485 - root - INFO -  Memory usage after forward pass: 33.46697998046875 GB.
2026-11-12 03:21:22,933 - root - INFO - Time taken for epoch 3 is 31.501408 sec, avg 8.126621 samples/sec
2026-11-12 03:21:22,944 - root - INFO -   Avg train loss=0.116889
2026-11-12 03:21:22,954 - root - INFO -   Avg val loss=0.14314872026443481
2026-11-12 03:21:22,959 - root - INFO -   Total validation time: 1.21921706199646 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 33.467 GB peak after the forward pass.
- Throughput: 8.13 samples/sec across 16 GPUs, with local batch 1.

The odd blocks roll the local grid by half a window along `sp1` and exchange the cells that cross the rank boundary. `sp2` has one rank, so that roll stays on the GPU. In the profile the NCCL kernels for the exchange sit on the default stream between the BF16 GEMMs. One step is about 485 ms, and the forward inside it is about 152 ms.

![Swin spatial profile, NCCL exchange for the shifted window during the forward](../../tutorial_images/swin_mp.png)
