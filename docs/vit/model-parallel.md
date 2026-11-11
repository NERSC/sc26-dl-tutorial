# ViT: model parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md)

Data parallelism still stores a full copy of the model and of the image on every GPU. At some point, as we grow the models, a local batch size of one also will not fit. `short_mp` is the next size up, `embed_dim: 1536` and `depth: 16`, with the same `patch_size: 16` and the same local batch of 1 (`global_batch_size: 4` on 4 GPUs). That step runs out of memory. Lowering the batch has nowhere left to go, so the replica itself has to be split. This is the typical way into model parallelism.

```bash
sbatch -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpus-per-task 24 \
    ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp --run_num oom \
    --tp=1 --sp1=1 --sp2=1
```

The model is built before the step fails. The log reports a local parameter count of 555.545 M.

```text
2026-11-09 21:03:09,311 - root - INFO - Local parameter count: 555.545 M
```

The forward then fails in `F.linear` inside the pointwise linear:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 768.00 MiB. GPU 1 has a total capacity of 39.49 GiB of which 760.00 MiB is free. Including non-PyTorch memory, this process has 38.74 GiB memory in use. Of the allocated memory 33.60 GiB is allocated by PyTorch, and 73.96 MiB is reserved by PyTorch but unallocated.
```

Hence, we need to shard the model. The sharding here follows the Megatron pattern: column-parallel and row-parallel linear layers, plus a spatial split of the image. It is a blueprint for parallelizing a scientific model in PyTorch, not a fully tuned stack. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) is the production version of the same ideas, including [context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) for a 1D token sequence. [`train_mp.py`](../../train_mp.py) builds the distributed module for the same four networks.

| Model | Distributed code | What is split |
|---|---|---|
| ViT | [`networks/vit_mp.py`](../../networks/vit_mp.py), [`networks/vit_te.py`](../../networks/vit_te.py) | Token sequence over `sp1 × sp2`. Heads and the MLP hidden size over `tp`. The TE file is the same mesh from Transformer Engine layers. |
| Swin | [`networks/swin_mp.py`](../../networks/swin_mp.py) | A block of windows on each spatial rank. Heads over `tp`. The window shift exchanges cells across the rank boundary. |
| FNO | [`networks/fno_mp.py`](../../networks/fno_mp.py) | Height over `sp1`, width over `sp2`. The FFT transposes between those axes. `--tp=1`. |
| ConvNeXt | [`networks/convnext_mp.py`](../../networks/convnext_mp.py) | Weights stay replicated. Ranks exchange a halo before the depthwise convolution. `--tp=1`. |

## Communicator groups

Treat the allocation as an `M × D` grid. `D` GPUs replicate the model, as on the [data-parallel page](data-parallel.md), and `M` GPUs split one replica. [`utils/comm.py`](../../utils/comm.py) factors `M` further, into tensor parallelism and two spatial axes:

```text
world size = dp × tp × sp1 × sp2
```

`dp` is orthogonal to the model-parallel axes. The ViT folds `sp1 × sp2` into one context-parallel group, `sp1-sp2`. Swin, the FNO, and ConvNeXt keep `sp1` and `sp2` as separate spatial axes.

`comm.init` asks [`utils/rank_generator.py`](../../utils/rank_generator.py) for the rank list of each group, then registers that list with `torch.distributed.new_group`. A collective takes the group it should run on:

```python
torch.distributed.all_reduce(tensor, group=comm.get_group("tp"))
```

Ranks absent from a list are not members of that group. `comm.get_group`, `comm.get_rank`, and `comm.get_size` are the lookups the layers use.

`order` chooses which index varies fastest. The default `tp-sp1-sp2-dp` makes tensor-parallel ranks a contiguous block, so they share a node. [`utils/check_rank_generator.ipynb`](../../utils/check_rank_generator.ipynb) prints the lists for any mesh.

Eight GPUs with `tp=4` and `dp=2` give tensor-parallel groups `[0, 1, 2, 3]` and `[4, 5, 6, 7]`, and data-parallel groups `[0, 4]`, `[1, 5]`, `[2, 6]`, `[3, 7]`. Rank 5 exchanges activations inside `[4, 5, 6, 7]` and reduces data-parallel gradients with rank 1.

Only `dp` changes which sample a rank reads. [`train_mp.py`](../../train_mp.py) sizes the loader from that group, so a model-parallel group trains on one example and each rank keeps its piece of it:

```python
params.data_num_shards = comm.get_size("dp")
params.data_shard_id = comm.get_rank("dp")
```

## Autograd mappings

A layer that communicated in the forward has to communicate the matching quantity in the backward. PyTorch will not infer that pair. Each one is a `torch.autograd.Function` in [`distributed/mappings.py`](../../distributed/mappings.py): `forward` runs one collective, `backward` runs its adjoint. The modules call the wrappers (`identity_forward_allreduce_backward`, and the others below) rather than `all_reduce` directly, so both directions stay defined.

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

## Transformer Engine

When the network really is a stack of attention and MLP blocks on a 1D token sequence, [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html) can own the sharding. The ViT flattens its patches into that kind of sequence, so [`networks/vit_te.py`](../../networks/vit_te.py) is a thin wrapper: pass the tensor-parallel group, and say whether the layer is column-parallel or row-parallel. TE shards the weight, inserts the matching collective, and fuses the epilogue.

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

The [Linear signature](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html) lists the rest of the arguments. Self-attention is `te.DotProductAttention`. The heads are the tensor-parallel axis. See the call in [`networks/vit_te.py`](../../networks/vit_te.py). The [attention signature](https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html) is the reference.

Swin, the FNO, and ConvNeXt do not look like that language-model block, so they stay on the custom modules above. Set `model_backend: transformer-engine` to select the ViT path. TE's context parallelism wants an even local sequence, and `vit_te.py` pads when the `sp1 × sp2` split is uneven. In a profile, TE's own NVTX ranges wrap each linear and the all-reduce that follows the row-parallel one. The column-parallel linear has no forward collective.

## Running a model-parallel job

[`submit_mp.sh`](../../submit_mp.sh) runs [`train_mp.py`](../../train_mp.py) with eight tasks per node, one per GPU. The single-node examples override this to four tasks on four GPUs. The mesh flags are the same for every network. `dp` is whatever GPUs remain.

```text
--tp      tensor-parallel size
--sp1     first spatial axis
--sp2     second spatial axis
--order   rank order, default tp-sp1-sp2-dp
```

```bash
sbatch --nodes=<nodes> ./submit_mp.sh \
    --yaml_config ./config/<file>.yaml --config <name> \
    --tp=1 --sp1=4 --sp2=1
```

For the ViT that ran out of memory, use 2 nodes. `sp1=4` shards the token sequence inside each node, and the global batch of 4 is split over `dp=4`, so the local batch stays 1.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp \
sbatch -N 2 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp --run_num mp \
    --tp=1 --sp1=4 --sp2=1
```

`sp1=4` brings the local parameter count to 480.048 M. The first three epochs stay under 40 GB. Epoch 4 is the profiled epoch.

```text
2026-11-09 21:13:17,782 - root - INFO - Starting Training Loop...
2026-11-09 21:13:24,538 - root - INFO -  Memory usage after forward pass: 33.72662353515625 GB.
2026-11-09 21:14:04,172 - root - INFO - Time taken for epoch 1 is 38.818769 sec, avg 1.545644 samples/sec
2026-11-09 21:14:04,177 - root - INFO -   Avg train loss=0.606727
2026-11-09 21:14:04,188 - root - INFO -   Avg val loss=0.3623535633087158
2026-11-09 21:14:04,193 - root - INFO -   Total validation time: 1.284818410873413 sec
2026-11-09 21:14:06,896 - root - INFO -  Memory usage after forward pass: 35.79498291015625 GB.
2026-11-09 21:14:49,885 - root - INFO - Time taken for epoch 2 is 40.914987 sec, avg 1.564219 samples/sec
2026-11-09 21:14:49,891 - root - INFO -   Avg train loss=0.304982
2026-11-09 21:14:49,901 - root - INFO -   Avg val loss=0.3405551314353943
2026-11-09 21:14:49,911 - root - INFO -   Total validation time: 1.9484591484069824 sec
2026-11-09 21:14:52,629 - root - INFO -  Memory usage after forward pass: 36.04498291015625 GB.
2026-11-09 21:15:35,487 - root - INFO - Time taken for epoch 3 is 41.071731 sec, avg 1.558249 samples/sec
2026-11-09 21:15:35,492 - root - INFO -   Avg train loss=0.292403
2026-11-09 21:15:35,502 - root - INFO -   Avg val loss=0.3354734778404236
2026-11-09 21:15:35,507 - root - INFO -   Total validation time: 1.9264698028564453 sec
```

Recorded performance, leaving out the profiled epoch:

- GPU memory: 36.045 GB peak after the forward pass.
- Throughput: 1.56 samples/sec across 16 GPUs, with local batch 1.

Attention keeps the query on this rank's tokens and all-gathers the keys and values over `sp1-sp2` before the score. In the profile that gather is `ncclDevKernel_AllGather` on the default stream, about 874 µs, between the BF16 GEMMs of the forward.

![ViT spatial profile, NCCL all-gather of keys and values in the forward](../../tutorial_images/vit_mp.png)

`short_mp_te` is that same model with `model_backend: transformer-engine`. The mesh is unchanged: `sp1=4` shards the sequence inside the node, and `dp=4` keeps the local batch at 1. `NVTE_BATCH_MHA_P2P_COMM=1` batches the context-parallel K/V send and recv.

```bash
NVTE_BATCH_MHA_P2P_COMM=1 \
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp_te \
sbatch -N 2 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_te --run_num mp_te \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp_te/4MP/mp_te/out.log`. Profiles: `vit_mp_te.0.nsys-rep` through `vit_mp_te.15.nsys-rep`.

![ViT Transformer Engine profile, K/V send/recv overlapped with flash attention](../../tutorial_images/vit_te_sp.png)

The same config with tensor parallelism leaves the sequence whole. `tp=4` shards the heads and the MLP across four GPUs within a node. `1536` and the 16 heads divide by 4. Two nodes leave `dp=4`, so the local batch stays 1. `CUDA_DEVICE_MAX_CONNECTIONS=1` keeps the backward all-reduce ahead of the weight-gradient GEMM so the two overlap.

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp_te_tp \
sbatch -N 2 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_te --run_num mp_te_tp \
    --tp=4 --sp1=1 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp_te/4MP/mp_te_tp/out.log`. Profiles: `vit_mp_te_tp.0.nsys-rep` through `vit_mp_te_tp.15.nsys-rep`.

![ViT Transformer Engine profile, backward all-reduce overlapped with the weight-gradient GEMM](../../tutorial_images/vit_te_tp.png)

Look for that overlap in the two profiles. In the sequence-sharded run it is the K/V P2P (`ncclDevKernel_SendRecv`) beside the flash-attention kernel. In the tensor-parallel run it is the backward all-reduce of the input gradient beside the weight-gradient GEMM.


A few sizes have to divide evenly. Swin's `window_size` has to divide the patch grid on each spatial rank. The FNO's channel count has to be at least `sp1` and at least `sp2`. Transformer Engine wants an even local sequence, and [`networks/vit_te.py`](../../networks/vit_te.py) pads when `model_backend: transformer-engine` and the split is uneven. With ConvNeXt local reads, keep `sp2=1` so each rank reads whole rows.


> **TODO:** Nsight Systems timeline zoomed to the activation collectives in forward and backward.

*Which activations does a spatial split leave replicated on every rank? Which weights does tensor parallelism leave replicated? On this model, which of those two splits moves less data per step?*

Picking `dp`, `tp`, and `sp1`/`sp2`, and which of them sits on the fast links inside a node, depends on how much each collective moves and on the machine. The usual tool for that choice is a performance model. [Performance models for transformers in science](https://arxiv.org/abs/2410.00273) works through one.

## Activation checkpointing

Activation checkpointing is an alternative that leaves the replica whole. `short_mp_ckpt` is the out-of-memory model with `activation_checkpoint: true`. Each block keeps its input, and the backward recomputes the block. One node, with tensor and spatial parallelism off:

```bash
sbatch -N 1 --ntasks-per-node 4 --gpus-per-node 4 --cpus-per-task 24 \
    ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_ckpt --run_num ckpt \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp_ckpt/1MP/ckpt/out.log`. Profiling stays off.

The step is much slower, because every block runs again in the backward. A slower step can also cost more GPU hours than splitting the same model across more GPUs. There is a floor. The checkpoint still stores the weights, the input to each block, and the activations of the block being recomputed. Grow the model past that remainder and checkpointing cannot make the step fit.

*Try this run. What do you see for memory after the forward, and for samples per second?*

## CUDA graphs (optional)

[`train_mp_graphs.py`](../../train_mp_graphs.py) captures the training step in a CUDA graph and replays it. A graph removes the CPU launch gap between kernels. That gap matters when a step is many small launches, and when several communicators have to stay in lockstep: the CPU jitter shows up as ranks waiting on each other. [`submit_mp.sh`](../../submit_mp.sh) calls [`train_mp.py`](../../train_mp.py). Compare the two scripts for where the graph is captured and where it is replayed.
