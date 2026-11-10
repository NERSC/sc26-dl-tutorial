# Deep Learning at Scale - Engineering High Resolution Scientific AI models (SC26)

The tutorial trains image-to-image models on high-resolution fields: the same kind of data used in scientific applications such as weather and climate, fluid dynamics, seismology, cosmology, and experimental imaging. Every run in this repository uses 4096 × 4096 inputs from an example turbulence simulation. We start from a single GPU, speed up data loading and compute, then scale with data parallelism and with model parallelism across tensor and spatial dimensions.

We focus on four architectures that represent the general architecture space of scientific AI models. These are Vision Transformer (ViT) based for global mixing, Local attention based models like Swin Transformer for local mixing, Fourier Neural Operators (FNO) for spectral mixing, and ConvNeXt for convolution based local mixing. Since each architecture class mixes information in different ways, we will see how this affects performance and scaling. 

![High-resolution fields from weather, fluids, and experimental imaging](tutorial_images/domains.png)

![ViT, Swin, FNO, and ConvNeXt](tutorial_images/models.png)

**Contents**

* [Links](#links)
* [Setup](#setup)
* [Single-GPU training](#single-gpu-training)
* [Optimizing GPU training](#optimizing-gpu-training)
* [Data parallelism](#data-parallelism)
* [Model parallelism](#model-parallelism)

## Links

Tutorial slides: https://drive.google.com/drive/folders/14OJ1mMEzsYa1MC2G2jfXJZpbE_HxoKzh?usp=sharing

Join the Slack workspace:

## Setup

First, complete the following steps:

1. **Clone the repository**

```bash
git clone https://github.com/NERSC/sc26-dl-tutorial.git
cd sc26-dl-tutorial
```

2. **Install Nsight Systems**

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

The inputs are an example turbulence simulation: HDF5 trajectories of 4096 × 4096 fields with 2 channels. [`utils/data_loader.py`](utils/data_loader.py) serves them.

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
| [`config/FNO.yaml`](config/FNO.yaml) | `short` (and `base`). `embed_dim` stays small: the blocks store activations on the full 4096 × 4096 grid. |
| [`config/ConvNeXt.yaml`](config/ConvNeXt.yaml) | `small` |

The same files will grow a matching block for the optimized loader, for data parallelism, and for spatial parallelism. Later sections point at those blocks as they are filled in.

Copy a block to change a model without editing the baseline:

```yaml
# in config/ViT.yaml
my_vit:
  <<: *base
  embed_dim: 768
  num_iters: 80
```

### Submit a job and view the results

**Submit a single-GPU job.** [`submit_pm.sh`](submit_pm.sh) starts the tutorial container and runs [`train.py`](train.py) with one GPU per task. The script mounts the dataset and the log directory.

```bash
sbatch -n 1 ./submit_pm.sh --yaml_config ./config/ViT.yaml --config short
```

The copied block from above is the same script with a different `--config`:

```bash
sbatch -n 1 ./submit_pm.sh --yaml_config ./config/ViT.yaml --config my_vit
```

Arguments after the script name are forwarded to `train.py`. Slurm writes the job output to a `.out` file beside the script. `squeue --me` shows the job id.

`expdir` (default `/logs`) is the log path inside the container. The run is written to `<expdir>/<config>/<N>GPU/<run_num>/`.

Open [`start_tensorboard.ipynb`](start_tensorboard.ipynb) on `LOGDIR` to plot the scalars.

> **TODO:** baseline training curve (loss, validation RMSE, samples/sec).

## Optimizing GPU training

The baseline leaves the GPU idle while a 4096 × 4096 batch is read, and it runs the step in FP32. Profile that run first, then overlap the load with the step, then speed up the math. The knobs below are keys in the YAML block. [`config/ViT.yaml`](config/ViT.yaml) gathers them into `short_opt` (workers, BF16, fused Adam, `torch.compile`) and `short_opt_dali` (the same block with the DALI loader). The other network files will get the same pair.

### Profiling

`train.py` already wraps the host-to-device copy, the forward, and the optimizer step in NVTX ranges (`torch.cuda.nvtx.range`), and it calls `torch.cuda.profiler.start()` and `stop()` around the fourth epoch. The ranges are what label the timeline. The profiler calls are what limit the capture to that epoch, so a profiled config needs at least four epochs. Grep `train.py` for `nvtx` and `profiler` to see the calls.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=baseline sbatch -n 1 ./submit_pm.sh --yaml_config ./config/ViT.yaml --config short
```

The script wraps the process in `nsys profile` and writes `baseline.<rank>.nsys-rep` under the log directory. `--trace=cuda,nvtx` keeps the trace to CUDA and the ranges we added. `-c cudaProfilerApi` tells Nsight to record only the stretch between `profiler.start` and `profiler.stop`. Copy the report back and open it in the Nsight Systems install from [Setup](#setup). The same prefix works on the later job scripts.

Zoomed out, the timeline has idle gaps between steps. Those gaps are the data load, which the next section closes. Zoomed into one step, the same view shows where the GPU time inside the step goes.

> **TODO:** Nsight Systems timeline for the baseline, zoomed out and for one iteration.

### The PyTorch data loader

[`utils/data_loader.py`](utils/data_loader.py) already sets `pin_memory`. That stages the batch in pinned host memory, which raises the bandwidth of the copy onto the GPU.

The knob left open is `num_data_workers`. At `0`, which is also PyTorch's default, the training process itself reads the next batch. On a 4096 × 4096 field that read is long enough to show up as the gap in the baseline profile. A value above 0 forks worker processes that read ahead of the step and hide that cost. A few workers, typically somewhere from 2 to 8, is enough. Past that the workers contend with the training process and throughput turns down.

Workers remove the gap between steps. The copy of the batch onto the GPU is still on the same CUDA stream as the compute, so it still sits at the front of the step. Prefetching that copy on its own stream is possible in plain PyTorch. A worked example is the [prefetching loader in NVIDIA's ConvNet examples](https://github.com/NVIDIA/DeepLearningExamples/blob/41f582bd9f65f6ebede77532b7cd64f038a8a380/PyTorch/Classification/ConvNets/image_classification/dataloaders.py#L354). DALI does that overlap for us.

> **TODO:** timeline with workers enabled. The gaps between steps should be gone, and the host-to-device copy should still lead the step.

### NVIDIA DALI

Two costs in the PyTorch loader stay even after the worker count is tuned.

1. I/O and any augmentation run as CPU work.
2. The workers are processes. Spawning and talking to them costs more than threads.

[DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) is the loader in [`utils/data_loader_dali.py`](utils/data_loader_dali.py), selected with `data_loader_config: dali`.

1. Its pipeline can run augmentation on the GPU.
2. Its workers are threads in the C++ runtime, so the prefetch keeps up with the step.

In the timeline, DALI shows up as extra CUDA streams beside the PyTorch stream. Those streams move the next batch, and run DALI's own kernels, while the current step is still computing. That overlap is the gain over the native loader on a field this large.

```bash
sbatch -n 1 ./submit_pm.sh --yaml_config ./config/ViT.yaml --config short_opt --run_num=opt
ENABLE_PROFILING=1 PROFILE_OUTPUT=dali sbatch -n 1 ./submit_pm.sh --yaml_config ./config/ViT.yaml --config short_opt_dali --run_num=dali
```

> **TODO:** throughput for `short_opt` and `short_opt_dali`, and the DALI timeline with its extra streams beside the training step.

### Mixed precision

With the load out of the way, the step itself is still FP32. Automatic mixed precision (AMP) runs the eligible ops in FP16 or BF16 and keeps the rest in FP32, which turns on tensor cores.

`train.py` uses two pieces from `torch.amp`. `autocast` picks the dtype of each op. `GradScaler` scales the loss so FP16 gradients, which have a narrow range, do not underflow. BF16 has the same exponent range as FP32, so BF16 needs no loss scaling. `amp_mode: bf16` is the setting used here. `amp_mode: fp16` is the path that turns the scaler on. The FNO's spectral weights are complex, and the scaler cannot unscale them, so the FNO configs stay on BF16.

Tensor cores are already in use for FP32 GEMMs through TF32, which PyTorch enables by default for cuDNN and cuBLAS. `NVIDIA_TF32_OVERRIDE=0` turns TF32 off. The baseline should slow down. TF32 only changes the math inside the kernel. The activations and weights stay FP32, so the traffic to memory is unchanged. FP16 and BF16 cut that traffic as well as the math, which is why AMP still wins on top of TF32. The forward and the backward both shrink, because these networks spend their time in GEMMs.

> **TODO:** timeline with BF16. The forward range, and the backward with it, should be much shorter than in the FP32 profile.

### `torch.compile` and fused Adam

AMP does the large speedup. Two smaller ones are still worth turning on.

`enable_fused: true` selects the fused Adam implementation in `torch.optim.Adam`. That update used to live in [APEX](https://github.com/NVIDIA/apex). It is in PyTorch now. One kernel applies the moment updates and the weight write, instead of a chain of small kernels, which cuts launch latency and keeps the values in registers.

`enable_jit: true` wraps the model in `torch.compile`, which fuses the elementwise ops around those GEMMs.

Both are on in `short_opt`. Because the step is dominated by large GEMMs, fusing the surrounding kernels is a smaller win than AMP, and it still shows up.

The FNO block leaves `enable_fused` off. Fused Adam does not update its complex spectral weights.

> **TODO:** timeline with fused Adam and `torch.compile` on top of BF16.

## Data parallelism

The single-GPU step is in good shape. The next gain is more GPUs. Each GPU holds a full copy of the model and a different slice of the batch, and the GPUs stay in step by averaging gradients. PyTorch's `DistributedDataParallel` (DDP) does that with the NCCL backend. The [distributed package docs](https://pytorch.org/docs/stable/distributed.html) cover the collectives underneath it.

### How the job is wired

[`submit_pm_dp.sh`](submit_pm_dp.sh) runs [`train.py`](train.py) with four tasks on each node, one process per GPU. `-N` is the number of nodes, so one node is 4 GPUs and four nodes are 16.

```bash
sbatch -N <nodes> ./submit_pm_dp.sh --yaml_config ./config/ViT.yaml --config <name>
```

*Why one process per GPU, rather than one process per node that owns all four devices?*

DDP can form its process group in several ways. This code uses environment variables, which Slurm already knows. [`export_DDP_vars.sh`](export_DDP_vars.sh), sourced by the job script, publishes `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, and the master address and port. Near the bottom of `train.py`, `torch.distributed.init_process_group` reads those and brings NCCL up.

Each rank has to see its own slice of the global batch, of size `global_batch_size / world_size`. The slice is built by `DistributedSampler` in [`utils/data_loader.py`](utils/data_loader.py). Find that constructor and check which of `data_num_shards` and `data_shard_id` it receives.

After the network is built, `train.py` wraps it:

```python
model = DistributedDataParallel(model, device_ids=[local_rank])
```

The wrapper broadcasts the initial weights so every rank starts from the same point, and during `backward()` it all-reduces gradients in buckets so the communication overlaps the rest of the backward. The optimizer then sees one agreed-upon gradient.

*Why does DDP broadcast the initial weights? What would the replicas do if each rank kept the weights it initialized on its own?*

DDP's buckets are the reason the all-reduces show up in the middle of the backward rather than as one blob after it. The bucket size is `bucket_cap_mb` on the `DistributedDataParallel` constructor (PyTorch's default is 25 MB). A smaller bucket flushes more often, so the profile shows more, shorter, syncs.

*If the code skipped DDP and called `torch.distributed.all_reduce` on the gradients once, just before the optimizer step, where would that sync sit in the profile?*

### Scaling the batch

Hold the local batch fixed and let the global batch grow with the GPU count. A larger global batch is a less noisy gradient, so the usual companion is a larger learning rate. One rule of thumb is the square-root rule: multiply the single-GPU learning rate by `sqrt(global_batch_size / base_batch_size)`. The YAML does not apply that rule for you. `lr` in the block is the rate that runs. Copy the block and set another value.

*If the learning rate stays at the single-GPU value while the global batch grows, what happens to the loss curve?*

*If you raise the learning rate and leave the batch size alone, what do you expect instead?*

*As the GPU count grows, what happens to samples per second, and what happens to how fast the loss actually falls? Which scale finishes a fixed number of samples sooner, and which spends less GPU-time? Can you write a block that overfits, or one that diverges?*

Each network's file will have a data-parallel block of this shape. In [`config/ViT.yaml`](config/ViT.yaml) that block is `short_opt_dali_dp`: the optimized ViT, global batch 16 on 4 GPUs, so the local batch stays 4.

```bash
sbatch -N 1 ./submit_pm_dp.sh --yaml_config ./config/ViT.yaml --config short_opt_dali_dp --run_num=dp4
```

TensorBoard events for a `train.py` run land under `<expdir>/<config>/<N>GPU/<run_num>/logs/`. Switch the horizontal axis to relative time. Throughput shows up as how quickly a run moves across that axis. The loss and the RMSE show whether the larger batch is still training.

> **TODO:** samples/sec and the validation curve against relative time, for each model's data-parallel config.

Profile with the same `ENABLE_PROFILING=1` prefix. The report is captured on one rank unless you arrange otherwise. Find the gradient all-reduces and see whether they overlap compute in the backward.

> **TODO:** Nsight Systems timeline zoomed to the gradient synchronization.

## Model parallelism

Data parallelism stores a full copy of the model, and of the image, on every GPU. A wider network, or the same network on this 4096 × 4096 grid with smaller patches, raises both the activation memory and the weight memory. Past a point the step does not fit, and the model has to be sharded.

The sharding here follows the Megatron pattern: column-parallel and row-parallel linear layers, plus a spatial split of the image. It is a blueprint for parallelizing a scientific model in PyTorch, not a fully tuned stack. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) is the production version of the same ideas, including [context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) for a 1D token sequence. [`train_mp.py`](train_mp.py) builds the distributed module for the same four networks.

| Model | Distributed code | What is split |
|---|---|---|
| ViT | [`networks/vit_mp.py`](networks/vit_mp.py), [`networks/vit_te.py`](networks/vit_te.py) | Token sequence over `sp1 × sp2`. Heads and the MLP hidden size over `tp`. The TE file is the same mesh from Transformer Engine layers. |
| Swin | [`networks/swin_mp.py`](networks/swin_mp.py) | A block of windows on each spatial rank. Heads over `tp`. The window shift exchanges cells across the rank boundary. |
| FNO | [`networks/fno_mp.py`](networks/fno_mp.py) | Height over `sp1`, width over `sp2`. The FFT transposes between those axes. `--tp=1`. |
| ConvNeXt | [`networks/convnext_mp.py`](networks/convnext_mp.py) | Weights stay replicated. Ranks exchange a halo before the depthwise convolution. `--tp=1`. |

### Communicator groups

Treat the allocation as an `M × D` grid. `D` GPUs replicate the model, as in the data-parallel section, and `M` GPUs split one replica. [`utils/comm.py`](utils/comm.py) factors `M` further, into tensor parallelism and two spatial axes:

```text
world size = dp × tp × sp1 × sp2
```

`dp` is orthogonal to the model-parallel axes. The ViT folds `sp1 × sp2` into one context-parallel group, `sp1-sp2`. Swin, the FNO, and ConvNeXt keep `sp1` and `sp2` as separate spatial axes.

`comm.init` asks [`utils/rank_generator.py`](utils/rank_generator.py) for the rank list of each group, then registers that list with `torch.distributed.new_group`. A collective takes the group it should run on:

```python
torch.distributed.all_reduce(tensor, group=comm.get_group("tp"))
```

Ranks absent from a list are not members of that group. `comm.get_group`, `comm.get_rank`, and `comm.get_size` are the lookups the layers use.

`order` chooses which index varies fastest. The default `tp-sp1-sp2-dp` makes tensor-parallel ranks a contiguous block, so they share a node. [`utils/check_rank_generator.ipynb`](utils/check_rank_generator.ipynb) prints the lists for any mesh.

Eight GPUs with `tp=4` and `dp=2` give tensor-parallel groups `[0, 1, 2, 3]` and `[4, 5, 6, 7]`, and data-parallel groups `[0, 4]`, `[1, 5]`, `[2, 6]`, `[3, 7]`. Rank 5 exchanges activations inside `[4, 5, 6, 7]` and reduces data-parallel gradients with rank 1.

Only `dp` changes which sample a rank reads. [`train_mp.py`](train_mp.py) sizes the loader from that group, so a model-parallel group trains on one example and each rank keeps its piece of it:

```python
params.data_num_shards = comm.get_size("dp")
params.data_shard_id = comm.get_rank("dp")
```

### Autograd mappings

A layer that communicated in the forward has to communicate the matching quantity in the backward. PyTorch will not infer that pair. Each one is a `torch.autograd.Function` in [`distributed/mappings.py`](distributed/mappings.py): `forward` runs one collective, `backward` runs its adjoint. The modules call the wrappers (`identity_forward_allreduce_backward`, and the others below) rather than `all_reduce` directly, so both directions stay defined.

The ViT MLP is the Megatron split, as `ColumnParallelLinear` and `RowParallelLinear`.

A column-parallel layer stores a slice of the output features and sees the full input. The matmul is local, so the forward collective is an identity. On the way back, each rank holds only a partial input-gradient, and the full gradient is their sum. The layer calls `identity_forward_allreduce_backward(x, "tp")` on its input. Attention uses the same layer for the query, key, and value projections, which also splits the heads across `tp`.

A row-parallel layer stores a slice of the input features. Each matmul is a partial output and the real output is the sum, so the forward ends in `allreduce_forward_identity_backward(x, "tp")`. The output-gradient arriving in the backward is already complete, and that pass stays local. The attention projection and the second MLP matrix are row-parallel.

Sequence sharding is the same pattern on the token axis. Attention leaves the query on the local tokens and gathers keys and values over `sp1-sp2` with `allgather_forward_reducescatter_backward`. The backward reduce-scatters the key and value gradients onto the tokens that owned them. `allgather_forward_split_backward` and `split_forward_allgather_backward` are the gather/split pair for a tensor that is cut without a reduction.

The other networks call the same kind of function. The FNO's FFT uses `distributed_transpose`: an all-to-all that swaps which axis is sharded, with the inverse transpose in the backward. ConvNeXt uses `halo_forward_sum_backward`, which copies neighbor edges in and, on the way back, returns the halo gradients to the rank that owns those cells. Swin's window shift uses `roll_forward_reverseroll_backward`.

A parameter that is a slice or a copy across a group carries `comm_metadata`. `init_ddp_model_and_reduction_hooks` in the same file reads it when the model is wrapped in DDP, and adds an all-reduce for any weight gradient the local backward has not already summed. A column-parallel weight, updated from this rank's tokens and copied across the spatial group, is marked:

```python
self.weight.comm_metadata = {
    "sharded": ["tp", None],
    "shared": ["sp1-sp2"],
    "reduce": ["sp1-sp2"],
}
```

`sharded` is the axis of the slice, `shared` is a group holding an identical copy, and `reduce` is that extra all-reduce.

*If a weight is shared across a model-parallel group, what has to be true of its initialization, and of its gradient?*

Shared weights must be initialized identically on every rank that holds them. If the backward only saw this rank's tokens, the weight gradient is a partial sum and needs the extra all-reduce named in `reduce`. DDP's own all-reduce covers the data-parallel copies. It does not cover that spatial sum. `init_ddp_model_and_reduction_hooks` registers it.

### Transformer Engine

When the network really is a stack of attention and MLP blocks on a 1D token sequence, [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) can own the sharding. The ViT flattens its patches into that kind of sequence, so [`networks/vit_te.py`](networks/vit_te.py) is a thin wrapper: pass the tensor-parallel group, and say whether the layer is column-parallel or row-parallel. TE shards the weight, inserts the matching collective, and fuses the epilogue.

```python
self.fc1 = te.Linear(
    in_features,
    hidden_features,
    bias=True,
    sequence_parallel=False,
    tp_group=comm.get_group("tp"),
    tp_size=comm.get_size("tp"),
    parallel_mode="column",
    device=torch.cuda.current_device(),
)
```

The [Linear signature](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.Linear) lists the rest of the arguments. Self-attention is `te.DotProductAttention`. The heads are the tensor-parallel axis. See the call in [`networks/vit_te.py`](networks/vit_te.py). The [attention signature](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html#transformer_engine.pytorch.DotProductAttention) is the reference.

Swin, the FNO, and ConvNeXt do not look like that language-model block, so they stay on the custom modules above. Set `model_backend: transformer-engine` to select the ViT path. TE's context parallelism wants an even local sequence, and `vit_te.py` pads when the `sp1 × sp2` split is uneven. In a profile, TE's own NVTX ranges wrap each linear and the all-reduce that follows the row-parallel one. The column-parallel linear has no forward collective.

### Running a model-parallel job

[`submit_pm_mp.sh`](submit_pm_mp.sh) runs [`train_mp.py`](train_mp.py) with four tasks per node. The mesh flags are the same for every network. `dp` is whatever GPUs remain.

```text
--tp      tensor-parallel size
--sp1     first spatial axis
--sp2     second spatial axis
--order   rank order, default tp-sp1-sp2-dp
```

```bash
sbatch --nodes=<nodes> ./submit_pm_mp.sh \
    --yaml_config ./config/<file>.yaml --config <name> \
    --tp=1 --sp1=4 --sp2=1
```

Each YAML file will have a spatial-parallel block. `--order=sp1-tp-sp2-dp` keeps the spatial ranks adjacent when that axis is the wide one. Logs go to `<expdir>/<config>/<M>MP/<run_num>/`, with `M = tp × sp1 × sp2`.

A few sizes have to divide evenly. Swin's `window_size` has to divide the patch grid on each spatial rank. The FNO's channel count has to be at least `sp1` and at least `sp2`. Transformer Engine wants an even local sequence, and [`networks/vit_te.py`](networks/vit_te.py) pads when `model_backend: transformer-engine` and the split is uneven. With ConvNeXt local reads, keep `sp2=1` so each rank reads whole rows.

> **TODO:** memory after the forward pass, and samples/sec, for a spatial mesh and a tensor-parallel mesh.

> **TODO:** Nsight Systems timeline zoomed to the activation collectives in forward and backward.

*Which activations does a spatial split leave replicated on every rank? Which weights does tensor parallelism leave replicated? On this model, which of those two splits moves less data per step?*

Picking `dp`, `tp`, and `sp1`/`sp2`, and which of them sits on the fast links inside a node, depends on how much each collective moves and on the machine. The usual tool for that choice is a performance model. [Performance models for transformers in science](https://arxiv.org/abs/2410.00273) works through one.

### CUDA graphs

[`train_mp_graphs.py`](train_mp_graphs.py) captures the training step in a CUDA graph and replays it. A graph removes the CPU launch gap between kernels. That gap matters when a step is many small launches, and when several communicators have to stay in lockstep: the CPU jitter shows up as ranks waiting on each other. [`submit_pm_mp.sh`](submit_pm_mp.sh) calls [`train_mp.py`](train_mp.py). Compare the two scripts for where the graph is captured and where it is replayed.

> **TODO:** timeline comparing `train_mp.py` and `train_mp_graphs.py` on the same mesh.
