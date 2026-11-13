# Deep Learning at Scale - Engineering High Resolution Scientific AI models (SC26)

The tutorial trains image-to-image models on high-resolution fields: the same kind of data used in scientific applications such as weather and climate, fluid dynamics, seismology, cosmology, and experimental imaging. Every run in this repository uses 4096 × 4096 inputs from an example turbulence simulation. We start from a single GPU, speed up data loading and compute, then scale with data parallelism and with model parallelism across tensor and spatial dimensions.

We focus on four architectures that represent the general architecture space of scientific AI models. These are Vision Transformer (ViT) based for global mixing, Local attention based models like Swin Transformer for local mixing, Fourier Neural Operators (FNO) for spectral mixing, and ConvNeXt for convolution based local mixing. Since each architecture class mixes information in different ways, we will see how this affects performance and scaling. 

![High-resolution fields from weather, fluids, and experimental imaging](tutorial_images/domains.png)

![ViT, Swin, FNO, and ConvNeXt](tutorial_images/models.png)

**Contents**

* [Links](#links)
* [Setup](#setup)
* [Single-GPU training](#single-gpu-training)
* [Performance and scaling](#performance-and-scaling)

## Links

Tutorial slides: https://drive.google.com/drive/folders/14OJ1mMEzsYa1MC2G2jfXJZpbE_HxoKzh?usp=sharing

Join the Slack workspace: https://join.slack.com/t/nersc-dl-tutorial/shared_invite/zt-4glmxatya-tMbqGiAADXeUMa4owi4k1w

## Setup

### [Cluster setup](docs/cluster.md)

### Install Nsight Systems

Jobs write a profile on the compute node. Install [Nsight Systems](https://developer.nvidia.com/nsight-systems) on your own machine so you can open the `.nsys-rep` files.

* **Create an account.** The download needs a free NVIDIA developer login.
* **Download the host build.** On the [download page](https://developer.nvidia.com/gameworksdownload#?search=nsight%20systems), pick macOS, Windows, or the Linux `.rpm`, `.deb`, or `.run` package.
* **Install it locally.** Run that installer. Copy a profile back from `LOGDIR` and open it here.

## Single-GPU Training

### Models

[`train.py`](train.py) builds whichever serial network the config names. Each one maps the input field to a later field.

| Model | Mixing | Serial code |
|---|---|---|
| ViT | Attention across all patch tokens | [`networks/vit.py`](networks/vit.py) |
| Swin | Attention inside shifted windows | [`networks/swin.py`](networks/swin.py) |
| FNO | Fourier mixing across the field | [`networks/fno.py`](networks/fno.py) |
| ConvNeXt | Convolution over nearby cells, then a channel MLP | [`networks/convnext.py`](networks/convnext.py) |

* **ViT and Swin.** `embed_dim`, `depth`, and `patch_size`. The token count is `(4096 / patch_size)^2`. Swin also takes `window_size`.
* **FNO.** `embed_dim` is the hidden width. `modes_kh` and `modes_kw` are the wavenumbers kept after the FFT. `mlp_ratio` widens the channel MLP (`0` leaves it out). `downsample` runs the blocks on a coarser grid.
* **ConvNeXt.** `embed_dim`, `depth`, `patch_size`, and `kernel_size` for the depthwise convolution.

### Data

The data set is comprised of 64 independent trajectories of decaying 2D homogeneous isotropic turbulence. Each trajectory is simulated on a periodic 4096×4096 grid with no forcing and saved as 64 frames, 0.04 time units apart, from t=0 to 2.52. The simulations are initialized using a San-Staples style initialization (https://www.sciencedirect.com/science/article/pii/S0045793012001363)  with `kp = 32` and a randomized phase for each Fourier mode.  Each trajectory is one HDF5 file holding a float32 fields array of shape [64, 2, 4096, 4096] (time, u/v channels, y, x), cell-centered. This data is meant to be a simple example of high resolution data for the purpose of this tutorial. In practices, data sets can be much larger (many more snapshots and / or trajectories) and / or more complex (e.g. 3D, multiphysics, many more variables, etc.) [`utils/data_loader.py`](utils/data_loader.py) serves them to the training process.

* **Input and target.** A sample is two snapshots `dt` steps apart. The network sees the first and predicts the second.
* **Normalization.** Channel mean and standard deviation come from `stats_trajectory.h5`. Files are split 90 / 10 into training and validation.
* **The loader.** `SynDataset` plus a PyTorch `DataLoader`. `data_loader_config: pytorch` selects it. `num_data_workers: 0` reads the batch in the training process.

[`train.py`](train.py) then:

1. Build the network named in the config, Adam, and a mean-squared-error loss.
2. For each batch, copy it to the GPU, run the forward, call `backward()`, and update the weights.
3. Validate, and write TensorBoard scalars: training loss, validation loss, RMSE of the first channel, learning rate, and samples per second.

The learning rate follows a cosine decay over `num_iters` (`CosineAnnealingLR`). With `warmup` greater than 0, the rate rises linearly for that many steps and then follows the cosine.

### Configuration

One YAML file per network. A run is that file plus one named block:

```text
--yaml_config PATH    # which file, default ./config/ViT.yaml
--config NAME         # which block in that file, default base
```

* **Which network.** `network` inside the block. Omitted means the ViT. The others set `swin`, `fno`, or `convnext`.
* **Inheritance.** Blocks are anchored. A new block starts from another with `<<: *base` and lists only the keys it changes.
* **What lives in the file.** Image size and channel counts, `global_batch_size`, `lr`, `num_iters`, and the loader keys above. `limit_nsamples` and `limit_nsamples_val` shorten a run. `amp_mode`, `enable_fused`, and `enable_jit` stay off for the baseline.
* **Command line.** `--run_num` tags the output directory. `--expdir` overrides the log path for one job.

| File | Baseline block |
|---|---|
| [`config/ViT.yaml`](config/ViT.yaml) | `short` (and `base` for a long run) |
| [`config/swin.yaml`](config/swin.yaml) | `short` (and `base`) |
| [`config/FNO.yaml`](config/FNO.yaml) | `short` (and `base`). `downsample: 4` runs the blocks on a 1024×1024 latent grid. |
| [`config/ConvNeXt.yaml`](config/ConvNeXt.yaml) | `short` (and `base`) |

The same files will grow a matching block for the optimized loader, for data parallelism, and for spatial parallelism. The architecture pages point at those blocks as they are filled in.

Copy a block to change a model without editing the baseline:

```yaml
# in config/ViT.yaml
my_vit:
  <<: *base
  embed_dim: 768
  num_iters: 80
```

### Submit a job and view the results

**Example: submit a single-GPU job.** [`submit.sh`](submit.sh) starts the tutorial container and runs [`train.py`](train.py) with one GPU per task. The script mounts the dataset and the log directory. This command is only an example of the submission syntax; we will mark job submission scripts to be submitted explicitly in the later sections. 

```bash
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short
```

The copied block from above is the same script with a different `--config`:

```bash
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config my_vit
```

Arguments after the script name are forwarded to `train.py`. Slurm writes the job output to `<yaml>_<config>[_<split>]_<jobid>.out` in the directory you submitted from, for example `vit_short_<jobid>.out`. The split part names the `--tp`, `--sp1`, and `--sp2` values above 1, such as `_sp4` or `_tp4`. The job shows up under the same name in `squeue`. `squeue --me` shows the job id.

`expdir` (default `/logs`) is the log path inside the container. The run is written to `<expdir>/<config>/<N>GPU/<run_num>/`.

Let us begin. Start by submitting the baseline job; this will take several minutes to complete since it is unoptimized and running on a single GPU.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short --run_num short
```

The profile of that run, and the optimized runs after it, are on the [ViT single-GPU page](docs/vit/single-gpu.md).

## Performance and Scaling

The measured runs are written up per architecture. We have filled in ViT for you. We will go over the other architectures during the tutorial as well.

### 1. Optimize for a single GPU

One GPU, before adding any more.

- Profile the FP32 baseline and see where the GPU waits.
- Turn on workers, BF16, fused Adam, and `torch.compile` together.
- Move the loader to DALI.

[ViT](docs/vit/single-gpu.md) · [Swin](docs/swin/single-gpu.md) · [FNO](docs/fno/single-gpu.md) · [ConvNeXt](docs/convnext/single-gpu.md)

### 2. Scale with data parallelism until the local batch is 1

More GPUs, each still holding a full copy of the model.

- Hold the local batch fixed while the global batch grows.
- Drop the local batch to 1 as the model gets larger.
- Raise the learning rate with the global batch.

[ViT](docs/vit/data-parallel.md) · [Swin](docs/swin/data-parallel.md) · [FNO](docs/fno/data-parallel.md) · [ConvNeXt](docs/convnext/data-parallel.md)

### 3. Scale with model parallelism

Larger models and inputs, once a local batch of 1 will not fit on the GPU.

- Split one replica across GPUs, with tensor parallelism and spatial parallelism.
- Shard the sequence so the step fits again.

[ViT](docs/vit/model-parallel.md) · [Swin](docs/swin/model-parallel.md) · [FNO](docs/fno/model-parallel.md) · [ConvNeXt](docs/convnext/model-parallel.md)
