# ViT Model Analysis

This analysis fixes the `mp` configuration in `config/ViT.yaml` and uses
`train_mp.py` as the training contract.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Entry point | `train_mp.py -> networks.vit_mp.ViT(params)` |
| Inputs | `inp: (16, 4, 4096, 4096)` |
| Outputs | `(16, 4, 4096, 4096)` |
| Loss | Mean relative L2 norm, `l2_loss_opt` |
| Parameter dtype | FP32 |
| Activation/compute dtype | BF16 model autocast; FP32 loss difference |
| Optimizer-state dtype | FP32 |
| FP32 master weights | No separate copy; model parameters are FP32 |
| Trainable parameters (M) | 188.906 |
| Optimizer | Fused Adam, betas `(0.9, 0.95)`, learning rate `1e-3` |
| Checkpointing | None |
| Assumptions | Fused BF16 SDPA; global batch 16; decimal GB |

## Representative Model Configuration

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Global batch | 16 | `global_batch_size` |
| Input/output channels | 4 / 4 | `n_in_channels`, `n_out_channels` |
| Image size | `4096 x 4096` | `img_size` |
| Patch size | `64 x 64` | `patch_size` |
| Patch grid / sequence length | `64 x 64` / 4096 | Derived |
| Layers | 12 | `depth` |
| Hidden size | 1024 | `embed_dim` |
| Attention heads | 16 | `num_heads` |
| Head width | 64 | `embed_dim / num_heads` |
| MLP/intermediate size | 4096 | `mlp_ratio=4` |
| Reconstruction width per token | 16384 | `4 * 64 * 64` |
| Dropout | 0 | `dropout` |

Notation: `b=16`, `l=4096`, `e=1024`, `h=16`, `d=64`,
`f=4096`, and `o=16384`.

## Op Contracts

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| Patch convolution | 1 | `b 4 4096 4096` | `b l e` | Conv2d, kernel=stride=64 | Convolution dgrad and wgrad |
| Position add | 1 | `b l e`, `1 l e` | `b l e` | `X + P` | `dX=dY`; `dP=sum_b(dY)` |
| Block LayerNorm | 24 | `b l e` | `b l e` | Normalize over `e` | Local dgrad; affine wgrads reduce over `b,l` |
| Q/K/V projections | `12 x 3` | `b l e`, `e e` | `b h l d` | `b l e, e e -> b l e` | `dY W^T -> dX`; `X^T dY -> dW` |
| Attention core | 12 | Q/K/V: `b h l d` | `b h l d` | `softmax(QK^T/sqrt(d))V` | SDPA dQ, dK, and dV |
| Attention projection | 12 | `b l e`, `e e` | `b l e` | `b l e, e e -> b l e` | Linear dgrad and wgrad |
| MLP FC1 + GELU | 12 | `b l e`, `e f` | `b l f` | `b l e, e f -> b l f` | Linear dgrad/wgrad and GELU derivative |
| MLP FC2 | 12 | `b l f`, `f e` | `b l e` | `b l f, f e -> b l e` | Linear dgrad and wgrad |
| Final LayerNorm | 1 | `b l e` | `b l e` | Normalize over `e` | Local dgrad and affine wgrads |
| Reconstruction head | 1 | `b l e`, `e o` | `b 4 4096 4096` | Token linear followed by reshape | Linear dgrad and wgrad |
| Relative L2 loss | 1 | Prediction and target | Scalar | `mean(||p-t||_2 / ||t||_2)` | FP32 gradient with respect to prediction |

## Op Costs

Weights include biases. Cached activations are first-order retained-tensor
estimates without activation checkpointing.

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch convolution | 1 | 16.778 | 0.067 | 2.147 | 2.199 | 4.398 |
| Position embedding/add | 1 | 4.194 | 0.017 | 0.134 | Elementwise | Elementwise |
| Block LayerNorms | 24 | 0.049 | <0.001 | 3.221 | Elementwise | Elementwise |
| Q/K/V projections | 36 | 37.786 | 0.151 | 1.611 | 4.948 | 9.896 |
| Attention `QK^T` + `AV` | 12 | 0 | 0 | 6.442 | 13.194 | 26.388 |
| Attention projection | 12 | 12.595 | 0.050 | 1.611 | 1.649 | 3.299 |
| MLP FC1 | 12 | 50.381 | 0.202 | 1.611 | 6.597 | 13.194 |
| GELU | 12 | 0 | 0 | 6.442 | Elementwise | Elementwise |
| MLP FC2 | 12 | 50.344 | 0.201 | 6.442 | 6.597 | 13.194 |
| Final LayerNorm | 1 | 0.002 | <0.001 | 0.134 | Elementwise | Elementwise |
| Reconstruction head | 1 | 16.777 | 0.067 | 0.134 | 2.199 | 4.398 |
| Relative L2 loss | 1 | 0 | 0 | 4.295 | Not counted | Not counted |
| **Total** | | **188.906** | **0.756** | **34.091** | **37.383** | **74.767** |

The FLOP total includes convolution, linear layers, and attention matmuls.
Elementwise normalization, activation, reshape, and loss operations are
excluded. Backward FLOPs are estimated as twice forward FLOPs.

The activation estimate assumes fused SDPA does not retain a full attention
score matrix. A BF16 score matrix for one layer would be 8.590 GB, so falling
back to a non-memory-efficient SDPA implementation materially changes the
result.

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 0.756 | Parameter count | 188.906M FP32 parameters |
| Gradients | 0.756 | Parameter count | One FP32 gradient per parameter |
| Optimizer state | 1.511 | Parameter count | Two FP32 Adam tensors |
| Saved activations | 34.091 | `b`, `l`, `e`, `f`, `h` | BF16 model tensors and FP32 loss difference |
| **Total per GPU** | **37.114** | Batch and sequence | Before sharding |

This is not peak runtime memory. It excludes FP32 input/target residency,
temporary kernel workspaces, communication buffers, CUDA context, DALI
prefetching, and allocator overhead.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Saved activations | 34.091 GB versus 3.022 GB for weights, gradients, and Adam states | Batch with DP; sequence with CP; TP intermediates |
| Attention compute | 13.194 forward TFLOPs and quadratic scaling in `l=4096` | Heads with TP; query context with CP |
| MLP compute | 13.194 forward TFLOPs across FC1 and FC2 | Intermediate width `f` with TP |
| Unsharded model boundaries | Patch convolution and reconstruction head each cost 2.199 TFLOPs | Batch with DP; current implementation replicates these over TP/CP |
