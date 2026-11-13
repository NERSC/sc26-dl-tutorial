# FNO: single GPU

[Tutorial](../../README.md) · [Data parallel](data-parallel.md) · [Model parallel](model-parallel.md)

The FNO mixes each channel with the others in Fourier space. Fused Adam stays off, because it cannot update the complex spectral weights. The training budget matches the [ViT single-GPU page](../vit/single-gpu.md) on samples, 64 / 16. Global batch is 1 on every step of this page. A batch of 4 does not fit, including under BF16: the resample FFT keeps the lift in FP32, and one map at 16 channels on the 4096×4096 image is 4 GB. `embed_dim: 76`, `depth: 8`, and modes 32×32 come to 94.9M real parameters, about twice the ViT base at 48.3M. A complex spectral weight counts as two. The parameter count in the log counts it once and reports 47.554 M. `downsample: 4` runs the blocks on a 1024×1024 latent grid. The lift and the projection stay on the full image, at `lift_dim: 16`. [`config/FNO.yaml`](../../config/FNO.yaml) holds the blocks, and [`networks/fno.py`](../../networks/fno.py) is the model.

## Profiling

`train.py` wraps the host-to-device copy, the forward, and the optimizer step in the same NVTX ranges as the ViT. A profiled config still needs at least four epochs, because the capture is the fourth epoch.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short --run_num fno
```

Log: `$HOME/sc26-dl-tutorial/logs/short/1GPU/fno/out.log`. Profile: `fno_short.0.nsys-rep`.

Submit this job.

## Workers, BF16, and `torch.compile`

`short_opt` turns on two workers, BF16, and `torch.compile`. The loader is still PyTorch. Fused Adam stays off.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short_opt --run_num fno_opt
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt/1GPU/fno_opt/out.log`. Profile: `fno_short_opt.0.nsys-rep`.

Submit this job.

## NVIDIA DALI

`short_opt_dali` keeps those settings and switches the loader to DALI.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short_opt_dali --run_num fno_dali
```

Log: `$HOME/sc26-dl-tutorial/logs/short_opt_dali/1GPU/fno_dali/out.log`. Profile: `fno_short_opt_dali.0.nsys-rep`.

Submit this job.

*How do the FNO profiles look like compared to the other models? What was different?*

*Look for the FFTs and other layers in the profile*

The next page scales this step across GPUs: [Data parallelism](data-parallel.md).
