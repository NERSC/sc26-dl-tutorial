# Homogeneous Swin Model Analysis

This analysis fixes the `mp` profile that will be added to `config/swin.yaml`
and uses `train_mp.py` as the representative training contract. The model is a
single-resolution Swin: patch embedding, 12 alternating unshifted/shifted
window blocks, final normalization, and a per-patch reconstruction head. It
does not use patch merging, relative-position bias, or activation
checkpointing.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Entry point | `train_mp.py -> networks.swin_mp.Swin(params, spatial_shard)` |
| Forward signature | `forward(x)` |
| Inputs | `x: (16, 4, 4096, 4096)` globally |
| Outputs | `(16, 4, 4096, 4096)` globally |
| Loss | Mean squared error, `mse_loss` |
| Parameter dtype | FP32 |
| Activation/compute dtype | BF16 model autocast; FP32 loss difference |
| Optimizer-state dtype | FP32 |
| FP32 master weights | No separate copy; model parameters are FP32 |
| Trainable parameters | 188.906M |
| Optimizer | Fused Adam, betas `(0.9, 0.95)`, learning rate `1e-3` |
| Checkpointing | None |
| Dynamic dimensions | Batch may vary subject to the DP batch equation; image, patch, embedding, head, depth, and window dimensions are fixed for this analysis |
| Assumptions | Fused BF16 SDPA; global batch 16; decimal GB; dropout disabled |

The baseline input/output contract intentionally matches `networks/vit.py`.
The reference weather Swin's temporal dimension, coordinate encoder, optional
large skip, Transformer Engine path, and multistep wrapper are not part of this
minimal model.

## Representative Model Configuration

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Global batch | 16 | Existing ViT `mp` profile |
| Input/output channels | 4 / 4 | Existing ViT `mp` profile |
| Image size | `4096 x 4096` | Existing ViT `mp` profile |
| Patch size | `64 x 64` | Existing ViT `mp` profile |
| Patch grid / token count | `64 x 64` / 4096 | Derived |
| Layers | 12 | Existing ViT `mp` profile |
| Hidden size | 1024 | Existing ViT `mp` profile |
| Attention heads | 16 | Existing ViT `mp` profile |
| Head width | 64 | `1024 / 16` |
| MLP/intermediate size | 4096 | `mlp_ratio=4` |
| Window size / tokens per window | `16 x 16` / 256 | Requested |
| Windows per image | 16 | `(64 / 16) * (64 / 16)` |
| Shift on odd blocks | `8 x 8` patches | Half-window shift |
| Reconstruction width per token | 16384 | `4 * 64 * 64` |
| Dropout | 0 | Existing ViT `mp` profile |

Notation: `b=16`, patch-grid height and width `hp=wp=64`, `l=4096`,
`e=1024`, `nh=16`, `d=64`, `f=4096`, window area `a=256`, and
reconstruction width `o=16384`.

## Op Contracts

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| Patch convolution | 1 | `b 4 4096 4096` | `b hp wp e` | Conv2d, kernel=stride=64 | Convolution dgrad; `dW` contracts batch and patch-grid axes |
| Position add | 1 | `b hp wp e`, `1 hp wp e` | `b hp wp e` | `X + P` | `dX=dY`; `dP=sum_b(dY)` |
| Block LayerNorm | 24 | `b hp wp e` | `b hp wp e` | Normalize over `e` | Local dgrad; affine wgrads contract `b,hp,wp` |
| Distributed/cyclic roll | 12 forward rolls plus 12 reverse rolls | `b hp wp e` | `b hp wp e` | Shift odd blocks by `(-8,-8)` before windows and `(8,8)` after | Inverse roll of dgrad |
| Window partition/reverse | 12 | `b hp wp e` | `(b*16) a e` | Rearrange into 16 windows, then reverse | Inverse rearrangement |
| Q/K/V projections | `12 x 3` | `(b*16) a e`, `e e` | `(b*16) nh a d` | `... e, e e -> ... e` | `dY W^T -> dX`; `X^T dY -> dW` |
| Window attention core | 12 | Q/K/V: `(b*16) nh a d` | same | `softmax(QK^T/sqrt(d) + mask)V` | Fused SDPA dQ, dK, and dV |
| Attention projection | 12 | `(b*16) a e`, `e e` | `(b*16) a e` | `... e, e e -> ... e` | Linear dgrad and wgrad |
| MLP FC1 + GELU | 12 | `b l e`, `e f` | `b l f` | `b l e, e f -> b l f` | Linear dgrad/wgrad and GELU derivative |
| MLP FC2 | 12 | `b l f`, `f e` | `b l e` | `b l f, f e -> b l e` | Linear dgrad and wgrad |
| Final LayerNorm | 1 | `b hp wp e` | `b hp wp e` | Normalize over `e` | Local dgrad; affine wgrads contract `b,hp,wp` |
| Reconstruction head | 1 | `b hp wp e`, `e o` | `b 4 4096 4096` | Token linear followed by unpatch reshape | Linear dgrad and wgrad |
| MSE loss | 1 | Prediction and target | Scalar | `mean((p-t)^2)` | FP32 gradient with respect to prediction |

## Op Costs

Weights include biases. Cached activations are first-order retained-tensor
estimates without activation checkpointing.

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch convolution | 1 | 16.778 | 0.067 | 2.147 | 2.199 | 4.398 |
| Position embedding/add | 1 | 4.194 | 0.017 | 0.134 | Elementwise | Elementwise |
| Block LayerNorms | 24 | 0.049 | <0.001 | 3.221 | Elementwise | Elementwise |
| Q/K/V projections | 36 | 37.786 | 0.151 | 1.611 | 4.948 | 9.896 |
| Window `QK^T` + `AV` | 12 | 0 | 0 | 6.442 | 0.825 | 1.649 |
| Attention projection | 12 | 12.595 | 0.050 | 1.611 | 1.649 | 3.299 |
| MLP FC1 | 12 | 50.381 | 0.202 | 1.611 | 6.597 | 13.194 |
| GELU | 12 | 0 | 0 | 6.442 | Elementwise | Elementwise |
| MLP FC2 | 12 | 50.344 | 0.201 | 6.442 | 6.597 | 13.194 |
| Final LayerNorm | 1 | 0.002 | <0.001 | 0.134 | Elementwise | Elementwise |
| Reconstruction head | 1 | 16.777 | 0.067 | 0.134 | 2.199 | 4.398 |
| MSE loss | 1 | 0 | 0 | 4.295 | Not counted | Not counted |
| **Total** | | **188.906** | **0.756** | **34.091** | **25.014** | **50.029** |

For one block, window-attention matmuls cost
`4 * b * l * a * e`; over 12 blocks this is 0.825 TFLOPs. Global ViT
attention substitutes `l` for `a`, so the 16x16 windows reduce this part of
forward compute by `l/a = 16x`. Linear, convolution, normalization, reshape,
masking, and reconstruction semantics otherwise remain homogeneous across
all 12 blocks.

The activation estimate assumes fused SDPA does not retain an
`a x a` attention matrix. Materializing all BF16 window score matrices would
require about 0.537 GB per layer globally, rather than the 8.590 GB per layer
required by global attention. Six shifted-block FP32 masks add about 0.025 GB
of persistent non-parameter buffers globally and are not included in saved
activations.

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 0.756 | Parameter count | 188.906M FP32 parameters |
| Gradients | 0.756 | Parameter count | One FP32 gradient per parameter |
| Optimizer state | 1.511 | Parameter count | Two FP32 Adam tensors |
| Saved activations | 34.091 | `b`, `l`, `e`, `f`, `nh` | BF16 model tensors and FP32 loss difference |
| **Total per GPU** | **37.114** | Batch and spatial token count | Before sharding |

This is not peak runtime memory. It excludes FP32 input/target residency,
attention masks, temporary kernel workspaces, communication buffers, CUDA
context, data-loader prefetching, NCCL internals, allocator fragmentation, and
framework overhead.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Saved activations | 34.091 GB versus 3.022 GB for weights, gradients, and Adam states | Batch with DP; patch-grid height with SP1; patch-grid width with SP2; TP intermediates |
| MLP compute | 13.194 forward TFLOPs across FC1 and FC2 | Intermediate width `f` with TP |
| Patch/head compute | Patch convolution and reconstruction head each cost 2.199 TFLOPs | Batch with DP and spatial grid with SP1/SP2 |
| Shifted-window boundaries | Odd blocks require globally correct cyclic shifts across rank boundaries | Distributed rolls over SP1 and SP2 |

Pure DP with a smaller local batch is sufficient for 40 GB in the
first-order estimate and is likely the throughput-optimal baseline. The
requested TP, SP1, and SP2 dimensions are therefore an explicit functionality
and scaling target, not the minimum mesh required merely to fit this model.
