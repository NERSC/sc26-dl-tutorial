# ViT: single GPU

[Tutorial](../../README.md) · [Data parallel](data-parallel.md) · [Model parallel](model-parallel.md)

The baseline leaves the GPU idle while a 4096 × 4096 batch is read, and it runs the step in FP32. Profile that run first. The next run, `short_opt`, turns on data workers, BF16, fused Adam, and `torch.compile` together. `short_opt_dali` then swaps in the DALI loader. The knobs are keys in the YAML block. [`config/ViT.yaml`](../../config/ViT.yaml) gathers them, and the other network files will get the same pair.

## Profiling

`train.py` already wraps the host-to-device copy, the forward, and the optimizer step in NVTX ranges (`torch.cuda.nvtx.range`), and it calls `torch.cuda.profiler.start()` and `stop()` around the fourth epoch. The ranges are what label the timeline. The profiler calls are what limit the capture to that epoch, so a profiled config needs at least four epochs. Grep `train.py` for `nvtx` and `profiler` to see the calls.

This profiled baseline is the first job to run in the tutorial which should have been submitted after the setup:

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short --run_num short
```

Log: `$HOME/sc26-dl-tutorial/logs/short/1GPU/short/out.log`. Profile: `vit_short.0.nsys-rep`.

The first three epochs look like this:

```text
2026-11-09 19:10:11,025 - root - INFO - Starting Training Loop...
2026-11-09 19:10:28,394 - root - INFO -  Memory usage after forward pass: 28.73602294921875 GB.
2026-11-09 19:12:41,772 - root - INFO - Time taken for epoch 1 is 132.280377 sec, avg 0.483821 samples/sec
2026-11-09 19:12:41,778 - root - INFO -   Avg train loss=0.458729
2026-11-09 19:12:41,788 - root - INFO -   Avg val loss=0.3124627470970154
2026-11-09 19:12:41,798 - root - INFO -   Total validation time: 6.951178550720215 sec
2026-11-09 19:12:50,667 - root - INFO -  Memory usage after forward pass: 30.91571044921875 GB.
2026-11-09 19:15:04,029 - root - INFO - Time taken for epoch 2 is 131.329318 sec, avg 0.487325 samples/sec
2026-11-09 19:15:04,035 - root - INFO -   Avg train loss=0.182501
2026-11-09 19:15:04,045 - root - INFO -   Avg val loss=0.21413132548332214
2026-11-09 19:15:04,055 - root - INFO -   Total validation time: 6.9553797245025635 sec
2026-11-09 19:15:12,936 - root - INFO -  Memory usage after forward pass: 30.91571044921875 GB.
2026-11-09 19:17:26,309 - root - INFO - Time taken for epoch 3 is 131.337044 sec, avg 0.487296 samples/sec
2026-11-09 19:17:26,314 - root - INFO -   Avg train loss=0.133589
2026-11-09 19:17:26,325 - root - INFO -   Avg val loss=0.18265116214752197
2026-11-09 19:17:26,335 - root - INFO -   Total validation time: 6.954867839813232 sec
```

Recorded performance:

- GPU memory: 30.916 GB peak.
- Throughput: 0.486 samples/sec on average.

The script wraps the process in `nsys profile` and writes `vit_short.<rank>.nsys-rep` under the log directory. `--trace=cuda,nvtx` keeps the trace to CUDA and the ranges we added. `-c cudaProfilerApi` tells Nsight to record only the stretch between `profiler.start` and `profiler.stop`. Copy the report back and open it in the Nsight Systems install from [Setup](../../README.md#setup). The same prefix works on the later job scripts.

Zoomed out, the timeline has idle gaps between steps. Those gaps are the data load, which the next section closes.

![ViT baseline profile, zoomed out across the profiled epoch](../../tutorial_images/vit_short_timeline.png)

Zoomed into one step, the host read of the next batch finishes before the GPU work starts. The step itself is mostly the forward and the attention GEMMs.

![ViT baseline profile, one training step](../../tutorial_images/vit_short_step.png)

## Workers, BF16, fused Adam, and `torch.compile`

`short_opt` is the same short ViT with four changes turned on together.

[`utils/data_loader.py`](../../utils/data_loader.py) already sets `pin_memory`. That stages the batch in pinned host memory, which raises the bandwidth of the copy onto the GPU. The knob left open in the baseline is `num_data_workers`. At `0`, which is also PyTorch's default, the training process itself reads the next batch. On a 4096 × 4096 field that read is long enough to show up as the gap in the baseline profile. `short_opt` sets it to 2. Worker processes read ahead of the step and hide that cost. A few workers, typically somewhere from 2 to 8, is enough. Past that the workers contend with the training process and throughput turns down.

Workers remove the gap between steps. The copy of the batch onto the GPU is still on the same CUDA stream as the compute, so it still sits at the front of the step. Prefetching that copy on its own stream is possible in plain PyTorch. A worked example is the [prefetching loader in NVIDIA's ConvNet examples](https://github.com/NVIDIA/DeepLearningExamples/blob/41f582bd9f65f6ebede77532b7cd64f038a8a380/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L354). DALI does that overlap in the next section.

The baseline step is FP32. Automatic mixed precision (AMP) runs the eligible ops in FP16 or BF16 and keeps the rest in FP32, which turns on tensor cores. `train.py` uses two pieces from `torch.amp`. `autocast` picks the dtype of each op. `GradScaler` scales the loss so FP16 gradients, which have a narrow range, do not underflow. BF16 has the same exponent range as FP32, so BF16 needs no loss scaling. `amp_mode: bf16` is the setting used here. `amp_mode: fp16` is the path that turns the scaler on. The FNO's spectral weights are complex, and the scaler cannot unscale them, so the FNO configs stay on BF16.

Tensor cores are already in use for FP32 GEMMs through TF32, which PyTorch enables by default for cuDNN and cuBLAS. `NVIDIA_TF32_OVERRIDE=0` turns TF32 off. The baseline should slow down. TF32 only changes the math inside the kernel. The activations and weights stay FP32, so the traffic to memory is unchanged. FP16 and BF16 cut that traffic as well as the math, which is why AMP still wins on top of TF32. The forward and the backward both shrink, because these networks spend their time in GEMMs.

`enable_fused: true` selects the fused Adam implementation in `torch.optim.Adam`. That update used to live in [APEX](https://github.com/NVIDIA/apex). It is in PyTorch now. One kernel applies the moment updates and the weight write, instead of a chain of small kernels, which cuts launch latency and keeps the values in registers.

`enable_jit: true` wraps the model in `torch.compile`, which fuses the elementwise ops around those GEMMs. Because the step is dominated by large GEMMs, fusing the surrounding kernels is a smaller win than AMP, and it still shows up. The FNO block leaves `enable_fused` off. Fused Adam does not update its complex spectral weights.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short_opt --run_num short_opt
```

The first epoch pays for `torch.compile`. The next two are the steady rate:

```text
2026-11-09 19:39:18,590 - root - INFO - Starting Training Loop...
2026-11-09 19:39:55,033 - root - INFO -  Memory usage after forward pass: 18.87664794921875 GB.
2026-11-09 19:40:14,670 - root - INFO - Time taken for epoch 1 is 27.557684 sec, avg 2.322401 samples/sec
2026-11-09 19:40:14,676 - root - INFO -   Avg train loss=0.460745
2026-11-09 19:40:14,686 - root - INFO -   Avg val loss=0.31617942452430725
2026-11-09 19:40:14,696 - root - INFO -   Total validation time: 3.941683769226074 sec
2026-11-09 19:40:18,116 - root - INFO -  Memory usage after forward pass: 19.87664794921875 GB.
2026-11-09 19:40:37,275 - root - INFO - Time taken for epoch 2 is 17.045505 sec, avg 3.754655 samples/sec
2026-11-09 19:40:37,281 - root - INFO -   Avg train loss=0.184020
2026-11-09 19:40:37,291 - root - INFO -   Avg val loss=0.21097834408283234
2026-11-09 19:40:37,296 - root - INFO -   Total validation time: 3.4452438354492188 sec
2026-11-09 19:40:40,546 - root - INFO -  Memory usage after forward pass: 19.87664794921875 GB.
2026-11-09 19:40:59,949 - root - INFO - Time taken for epoch 3 is 16.763713 sec, avg 3.817770 samples/sec
2026-11-09 19:40:59,955 - root - INFO -   Avg train loss=0.131843
2026-11-09 19:40:59,965 - root - INFO -   Avg val loss=0.17809422314167023
2026-11-09 19:40:59,975 - root - INFO -   Total validation time: 3.759243965148926 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 19.877 GB peak, down from 30.916 GB in the FP32 baseline.
- Throughput: 3.80 samples/sec, up from 0.486 samples/sec.

We see a significant improvement in throughput, and a reduction in GPU memory usage compared to the FP32 baseline.

The profiled epoch is a run of steps a little under a second each. The forward is about 275 ms, against a step of about 7.7 s in the FP32 baseline. A short host-to-device copy still sits between steps.

![ViT short_opt profile, steps in the profiled epoch](../../tutorial_images/vit_short_opt.png)

Zooming in shows the data-loader workers. Their `data/train/getitem` ranges run while the GPU is still in the training step. The copy onto the GPU stays on the main process, at the front of the next step.

![ViT short_opt profile, data-loader workers overlapped with the training step](../../tutorial_images/vit_short_opt_zoomed.png)

## NVIDIA DALI

Two costs in the PyTorch loader stay even after the worker count is tuned.

1. I/O and any augmentation run as CPU work.
2. The workers are processes. Spawning and talking to them costs more than threads.

[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) is the loader in [`utils/data_loader_dali.py`](../../utils/data_loader_dali.py), selected with `data_loader_config: dali`.

1. Its pipeline can run augmentation on the GPU.
2. Its workers are threads in the C++ runtime, so the prefetch keeps up with the step.

In the timeline, DALI shows up as extra CUDA streams beside the PyTorch stream. Those streams move the next batch, and run DALI's own kernels, while the current step is still computing. That overlap is the gain over the native loader on a field this large. `short_opt_dali` keeps the BF16, fused Adam, and `torch.compile` settings from `short_opt`.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short_opt_dali --run_num dali
```

The first epoch still pays for `torch.compile`. The next two are the steady rate:

```text
2026-11-09 19:54:40,715 - root - INFO - Starting Training Loop...
2026-11-09 19:54:49,540 - root - INFO -  Memory usage after forward pass: 24.43133544921875 GB.
2026-11-09 19:55:06,093 - root - INFO - Time taken for epoch 1 is 14.548313 sec, avg 4.124189 samples/sec
2026-11-09 19:55:06,099 - root - INFO -   Avg train loss=0.477418
2026-11-09 19:55:06,109 - root - INFO -   Avg val loss=0.30858129262924194
2026-11-09 19:55:06,119 - root - INFO -   Total validation time: 1.5312886238098145 sec
2026-11-09 19:55:07,133 - root - INFO -  Memory usage after forward pass: 25.43133544921875 GB.
2026-11-09 19:55:25,007 - root - INFO - Time taken for epoch 2 is 14.603768 sec, avg 4.382431 samples/sec
2026-11-09 19:55:25,013 - root - INFO -   Avg train loss=0.191646
2026-11-09 19:55:25,018 - root - INFO -   Avg val loss=0.21187180280685425
2026-11-09 19:55:25,023 - root - INFO -   Total validation time: 2.0731003284454346 sec
2026-11-09 19:55:26,050 - root - INFO -  Memory usage after forward pass: 25.43524169921875 GB.
2026-11-09 19:55:43,322 - root - INFO - Time taken for epoch 3 is 14.622557 sec, avg 4.376800 samples/sec
2026-11-09 19:55:43,327 - root - INFO -   Avg train loss=0.132309
2026-11-09 19:55:43,337 - root - INFO -   Avg val loss=0.17815913259983063
2026-11-09 19:55:43,342 - root - INFO -   Total validation time: 1.3880062103271484 sec
```

Recorded performance, leaving out epoch 1 and the profiled epoch:

- GPU memory: 25.435 GB peak, up from 19.877 GB with the PyTorch loader. Before the first step the device is already at 7.715 GB, against 1.148 GB for `short_opt`. That increase is DALI's copy of the prefetched batches.
- Throughput: 4.38 samples/sec, up from 3.80 samples/sec.

DALI's copy and its own kernels run on a second CUDA stream while the default stream is still inside the current step. The forward is about 276 ms, and each step is about 870 ms.

![ViT DALI profile, DALI kernels overlapped with the training step](../../tutorial_images/vit_short_opt_dali.png)

The next page scales this step across GPUs: [Data parallelism](data-parallel.md).
