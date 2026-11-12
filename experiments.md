# ViT Experiments

This document describes the named experiments in `config/ViT.yaml`, why each
exists, and its effective ViT configuration.

All experiments use:

- Image size: `4096 x 4096`
- Input/output channels: `2 / 2`
- Attention heads: `16`, except `24` for `short_mp` and its variants
  (`embed_dim 1536 / 24 = head_dim 64`)
- Dropout: `0`
- Optimizer: Adam

The number of tokens is `(4096 / patch_size)^2`. For distributed runs:

```text
world_size = dp * tp * sp1 * sp2
local_batch_size = global_batch_size / dp
```

## Experiment Matrix

| Config | Purpose | ViT `(embed, depth, patch, tokens)` | Global batch | Precision | Data loader | Optimizations | Data limits |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| `base` | Unoptimized reference and long training run | `(512, 12, 32, 16,384)` | 4 | FP32 | PyTorch, 0 workers | None | Unlimited |
| `short` | Short single-GPU baseline profile showing FP32 compute and input stalls | `(512, 12, 32, 16,384)` | 4 | FP32 | PyTorch, 0 workers | None | 64 train / 16 validation, 80 iterations |
| `short_opt` | Show the effect of workers, BF16, compilation, and fused Adam while retaining the PyTorch loader | `(512, 12, 32, 16,384)` | 4 | BF16 | PyTorch, 2 workers | `torch.compile`, fused Adam | 64 / 16, 80 iterations |
| `short_opt_dali` | Optimized single-GPU profile and PyTorch-versus-DALI comparison | `(512, 12, 32, 16,384)` | 4 | BF16 | DALI, 2 workers | `torch.compile`, fused Adam | 64 / 16, 80 iterations |
| `short_opt_dali_dp` | Data-parallel scaling of the optimized small model | `(512, 12, 32, 16,384)` | 16 | BF16 | DALI, 2 workers | `torch.compile`, fused Adam | 64 / 16, 80 iterations |
| `short_opt_big` | Larger-token and wider ViT used to motivate distributed memory scaling | `(1024, 12, 16, 65,536)` | 16 | BF16 | DALI, 2 workers | `torch.compile`, fused Adam | 64 / 16, 80 iterations |
| `opt_big` | Longer version of `short_opt_big` | `(1024, 12, 16, 65,536)` | 16 | BF16 | DALI, 2 workers | `torch.compile`, fused Adam | 64 / 16, 30,000 iterations |
| `short_mp` | Large model for spatial/model-parallel profiling | `(1536, 16, 16, 65,536)` | 4 | BF16 | DALI, 2 workers | Fused Adam; compilation disabled | 64 / 16, 80 iterations |
| `short_mp_te` | Compare native model parallelism with Transformer Engine context parallelism | `(1536, 16, 16, 65,536)` | 4 | BF16 | DALI, 2 workers | Transformer Engine, fused Adam | 64 / 16, 80 iterations |

## Suggested Run Progression

| Stage | Config | GPUs and mesh | Local batch | What it demonstrates |
| --- | --- | --- | ---: | --- |
| Baseline | `short` | 1 GPU: `dp=1,tp=1,sp1=1,sp2=1` | 4 | FP32 compute, serial loading, and GPU idle gaps |
| Native optimization | `short_opt` | 1 GPU | 4 | BF16, workers, compilation, and fused optimizer |
| DALI optimization | `short_opt_dali` | 1 GPU | 4 | Prefetching and overlapping data movement |
| Data parallel | `short_opt_dali_dp` | 4 GPUs: `dp=4` | 4 | DDP gradient synchronization at fixed local batch |
| Large-model DP | `short_opt_big` | 16 GPUs: `dp=16` | 1 | A wider model and four times as many tokens |
| Spatial parallel | `short_mp` | 16 GPUs: `dp=4,sp1=4` | 1 | Sequence/spatial sharding and K/V communication |
| TE spatial parallel | `short_mp_te` | 16 GPUs: `dp=4,sp1=4` | 1 | Transformer Engine context-parallel implementation |

## Current `run.sh`

The current launcher runs:

```text
config = short_mp
world size = 16
dp = 4
tp = 1
sp1 = 4
sp2 = 1
local batch = 1
```

Nsight Systems captures epoch 4, including training, validation, and validation
sample plotting. It writes one profile per rank using the config name.

## Configuration Note

`opt_big` inherits `limit_nsamples: 64` and `limit_nsamples_val: 16` through
`short_opt_big`. Despite the “no samples limits” comment in the YAML, it is not
currently an unlimited-data run.
