# ConvNeXt: single GPU

[Tutorial](../../README.md) · [Data parallel](data-parallel.md) · [Spatial parallel](model-parallel.md)

ConvNeXt mixes nearby feature cells with a depthwise convolution, then mixes
channels with an MLP. The [serial model](../../networks/convnext.py) keeps the
same feature-grid resolution through all blocks. A patch stem turns each
4096×4096, two-channel field into a 128×128 grid; the head reconstructs the
two-channel field. This implementation uses zero padding at the field boundary.

The three stages in [ConvNeXt.yaml](../../config/ConvNeXt.yaml) use patch 32,
width 512, depth 12, kernel 7, batch 4, and Adam with learning rate `5e-4`.
Each run uses 64 training and 16 validation samples, with 80 iterations
scheduling five epochs.

## Profiling

`train.py` marks data loading, device transfers, the forward pass, and optimizer
updates. The ConvNeXt block adds `depthwise convolution` and `mlp` ranges.
The profiler captures the fourth epoch.

Start with `short`: FP32 and the PyTorch loader with no workers.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short --run_num convnext
```

Log: `$HOME/sc26-dl-tutorial/logs/short/1GPU/convnext/out.log`. Profile: `convnext_short.0.nsys-rep`.

Submit this job. Record throughput and memory use from the log, and inspect
the profile for gaps between GPU steps.

## Workers, BF16, fused Adam, and `torch.compile`

`short_opt` enables two loader workers, BF16 autocast, fused Adam, and
compilation. The loader is still PyTorch.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt --run_num convnext_opt
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt/1GPU/convnext_opt/out.log`. Profile: `convnext_short_opt.0.nsys-rep`.

Submit this job. Compare its throughput, memory use, and step timeline with
the FP32 baseline.

## NVIDIA DALI

`short_opt_dali` keeps the optimized model settings and switches the loader
to DALI with two workers.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_dali --run_num convnext_dali
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_dali/1GPU/convnext_dali/out.log`. Profile: `convnext_short_opt_dali.0.nsys-rep`.

Submit this job. Compare the data-loading gaps and GPU activity with the
optimized PyTorch run. For throughput comparisons, use epochs 2, 3, and 5
to leave out startup and the profiled epoch.

Continue with [data parallelism](data-parallel.md).
