# ConvNeXt Model Analysis

This analysis fixes the `big` configuration in `config/ConvNeXt.yaml` and uses
`train_mp.py` as the training contract.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Entry point | `train_mp.py -> networks.convnext_mp.ConvNeXt(params, spatial_shard)` |
| Inputs | `inp: (4, 2, 4096, 4096)` |
| Outputs | `(4, 2, 4096, 4096)` |
| Loss | Mean squared error, `mse_loss` |
| Parameter dtype | FP32 |
| Activation/compute dtype | BF16 model autocast; FP32 loss difference |
| Optimizer-state dtype | FP32 |
| FP32 master weights | No separate copy; model parameters are FP32 |
| Trainable parameters (M) | 304.996 |
| Optimizer | Fused Adam, betas `(0.9, 0.95)`, learning rate `5e-4` |
| Checkpointing | None |
| Assumptions | Compilation enabled; global batch 4; decimal GB |

## Representative Model Configuration

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Global batch | 4 | `global_batch_size` |
| Input/output channels | 2 / 2 | `n_in_channels`, `n_out_channels` |
| Image size | `4096 x 4096` | `img_size` |
| Patch size | `16 x 16` | `patch_size` |
| Feature grid / spatial positions | `256 x 256` / 65536 | Derived |
| Layers | 16 | `depth` |
| Hidden size | 1536 | `embed_dim` |
| MLP/intermediate size | 6144 | `4 * embed_dim` |
| Depthwise kernel | `7 x 7` | `kernel_size` |
| Reconstruction width per patch | 512 | `2 * 16 * 16` |
| Layer scale | `1e-6` | `layer_scale_init` |
| Dropout | 0 | `MLP` default |

Notation: `b=4`, `h=w=256`, `t=h*w=65536`, `e=1536`, `f=6144`,
and `o=512`. All blocks retain the same spatial resolution and channel width.

## Op Contracts

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| Patch convolution | 1 | `b 2 4096 4096` | `b e h w` | Conv2d, kernel=stride=16 | Convolution dgrad and wgrad |
| Stem LayerNorm | 1 | `b h w e` | `b h w e` | Normalize over `e` | Local dgrad; affine wgrads reduce over `b,h,w` |
| Depthwise convolution | 16 | `b e h w` | `b e h w` | Channelwise convolution, kernel=7, padding=3 | Neighboring dgrad contributions; wgrads reduce over `b,h,w` |
| Block LayerNorm | 16 | `b h w e` | `b h w e` | Normalize over `e` | Local dgrad; affine wgrads reduce over `b,h,w` |
| MLP FC1 + GELU | 16 | `b h w e`, `e f` | `b h w f` | `b h w e, e f -> b h w f` | Linear dgrad/wgrad and GELU derivative |
| MLP FC2 | 16 | `b h w f`, `f e` | `b h w e` | `b h w f, f e -> b h w e` | Linear dgrad and wgrad |
| Layer scale + residual | 16 | `b h w e`, `e` | `b e h w` | `R + gamma * U` | Residual identity; scale wgrad reduces over `b,h,w` |
| Final LayerNorm | 1 | `b h w e` | `b h w e` | Normalize over `e` | Local dgrad and affine wgrads |
| Reconstruction head | 1 | `b h w e`, `e o` | `b 2 4096 4096` | Patch linear followed by reshape | Linear dgrad and wgrad |
| MSE loss | 1 | Prediction and target | Scalar | `mean((p-t)^2)` | FP32 gradient with respect to prediction |

## Op Costs

Weights include biases. Cached activations are first-order retained-tensor
estimates without activation checkpointing.

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Patch stem | 1 | 0.788 | 0.003 | 0.268 | 0.412 | 0.825 |
| Stem LayerNorm | 1 | 0.003 | <0.001 | 1.613 | Elementwise | Elementwise |
| Depthwise convolution | 16 | 1.229 | 0.005 | 12.885 | 0.631 | 1.263 |
| Block LayerNorm | 16 | 0.049 | <0.001 | 25.803 | Elementwise | Elementwise |
| FC1 | 16 | 151.093 | 0.604 | 12.885 | 79.165 | 158.330 |
| GELU | 16 | 0 | 0 | 51.540 | Elementwise | Elementwise |
| FC2 | 16 | 151.020 | 0.604 | 51.540 | 79.165 | 158.330 |
| Layer scale + residual | 16 | 0.025 | <0.001 | 12.885 | Elementwise | Elementwise |
| Final LayerNorm | 1 | 0.003 | <0.001 | 1.613 | Elementwise | Elementwise |
| Head + unpatch | 1 | 0.786 | 0.003 | 0.805 | 0.412 | 0.825 |
| MSE | 1 | 0 | 0 | 0.537 | Not counted | Not counted |
| **Total** | | **304.996** | **1.220** | **172.373** | **159.786** | **319.571** |

The FLOP total includes convolution and linear layers. Elementwise
normalization, activation, layer scale, reshape, and loss operations are
excluded. Backward FLOPs are estimated as twice forward FLOPs.

The activation estimate budgets FP32 LayerNorm inputs and statistics, BF16
convolution and MLP tensors, and an FP32 loss difference. One BF16 hidden
field is 0.805 GB; an expanded MLP field is 3.221 GB. Residual adds require no
separate saved input. Compilation can fuse operations or recompute values,
changing the retained-tensor estimate.

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 1.220 | Parameter count | 304.996M FP32 parameters |
| Gradients | 1.220 | Parameter count | One FP32 gradient per parameter |
| Optimizer state | 2.440 | Parameter count | Two FP32 Adam tensors |
| Saved activations | 172.373 | `b`, `h`, `w`, `e`, `f`, depth | BF16 model tensors, FP32 LayerNorm inputs and loss difference |
| **Total per GPU** | **177.253** | Batch and spatial grid | Before sharding |

This is not peak runtime memory. It excludes FP32 input/target residency,
output residency, temporary kernel workspaces, autocast weight copies,
communication buffers, CUDA context, DALI prefetching, and allocator overhead.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Saved activations | 172.373 GB versus 4.880 GB for weights, gradients, and Adam states | Batch with DP; feature-grid height/width with SP1 x SP2 |
| MLP compute | 158.330 forward TFLOPs across FC1 and FC2 | Batch with DP; spatial positions with SP1 x SP2 |
| Spatial dependencies | Depthwise convolution requires neighboring feature cells | Three-cell halo exchange on each split axis |
| Data delivery | Local width reads stalled while full reads and local height reads sustained training | Height stripes preserve contiguous rows in the input layout |
