# FNO Model Analysis

This analysis follows the structure of [the ViT analysis](../vit/analysis.md). It fixes the
`mp` configuration in `config/FNO.yaml` and uses `train_mp.py` as the training contract. The
reference implementation is the serial `networks/fno.py`; the distributed one is
`networks/fno_mp.py`. All numbers are first-order estimates from closed-form formulas; the
parameter counts were checked against the instantiated model. Nothing here is measured on a
GPU cluster.

## Model Contract

| Field | Value |
| --- | --- |
| Framework | PyTorch |
| Entry point | `train_mp.py -> networks.fno_mp.FNO(params)` (serial: `train.py -> networks.fno.FNO`) |
| Inputs | `inp: (16, 4, 4096, 4096)` |
| Outputs | `(16, 4, 4096, 4096)` |
| Interface | Spatially sharded when model parallel: each rank loads, takes and returns its `(H/sp1, W/sp2)` shard; the loss and metrics reduce over the spatial group |
| Loss | Mean squared error, `mse_loss` (from PR #3): sum of squares on the shard, all-reduced over SP1-SP2, divided by the global element count |
| Parameter dtype | FP32, and **complex64 for the spectral weights** |
| Activation/compute dtype | BF16 model autocast, **except the spectral layers, which run in FP32 with autocast disabled**; FP32 loss difference |
| Optimizer-state dtype | FP32 (complex64 for the spectral weights' Adam state) |
| FP32 master weights | No separate copy |
| Trainable parameters | 4.200 M elements: 4,194,304 complex (spectral) + 6,068 real |
| Optimizer | Adam, betas `(0.9, 0.95)`, learning rate `1e-3`. **Not fused**: fused Adam rejects complex parameters (see [comparison.md](comparison.md)) |
| Checkpointing | None (activation checkpointing is not implemented) |
| Assumptions | Global batch 16; decimal GB; spectral layers save only the truncated spectrum for their backward |

## Representative Model Configuration

| Dimension | Fixed value | Source |
| --- | ---: | --- |
| Global batch | 16 | `global_batch_size` |
| Input/output channels | 4 / 4 | `n_in_channels`, `n_out_channels` |
| Grid | `4096 x 4096` (`T = 16,777,216` points) | `img_size` |
| Blocks | 4 | `depth` |
| Hidden channels | 16 | `embed_dim` (`mp` config, see the width sweep) |
| Retained modes | `64 x 64` (`kh x kw`), 32 positive + 32 negative in h | `modes_kh`, `modes_kw` |
| Channel MLP width | 32 | `mlp_ratio=2` |
| Dropout | 0 | none in the FNO |

Notation: `b=16`, `T=H*W=2^24`, `C=16`, `f=32`, `kh=kw=64`, `n=4` blocks.

The FNO operates at full resolution in every layer. A single hidden activation is `b*C*T`
elements, 512x the ViT's `b*l*e` tokens-times-width (the ViT patchifies 4096² down to 4096
tokens). That drives everything below.

## Op Contracts

| Op | Count | Input shapes | Output shapes | Forward contract | Backward contract |
| --- | ---: | --- | --- | --- | --- |
| Lift fc1 + GELU + fc2 | 1 | `b 4 H W` | `b C H W` | 1x1 conv `4->C`, GELU, 1x1 conv `C->C` | Conv dgrad/wgrad, GELU derivative |
| Spectral: `rfft2` (ortho) | 4 | `b C H W` fp32 | `b C H W/2+1` c64 | Real FFT over w, FFT over h | Adjoint transform (same cost) |
| Spectral: truncate | 4 | `b C H W/2+1` | `b C kh kw` | Keep 32+32 rows in h, 64 columns in w | Zero-fill |
| Spectral: mode mixing | 4 | `b C kh kw`, `C C kh kw` | `b C kh kw` | `bihw,iohw->bohw`, complex | dgrad: `W^H`; wgrad: per-mode `X^H dY` |
| Spectral: zero-fill + `ifft` | 4 | `b C kh kw` | `b C H kw` | Pad middle h modes with zeros, FFT over h | Adjoint |
| Spectral: self-conjugate zeroing | 4 | `b C H kw` | same | Zero imag of w=0 and (even width) w=W/2 columns | Masked gradient |
| Spectral: `irfft` | 4 | `b C H kw` c64 | `b C H W` fp32 | c2r over w, ortho | Adjoint |
| Skip 1x1 | 4 | `b C H W` | `b C H W` | `C C` pointwise linear | Linear dgrad/wgrad |
| Add + GELU | 4 | `b C H W` x2 | `b C H W` | `act(spectral(x) + skip(x))` | GELU derivative |
| Channel MLP fc1 + GELU + fc2 | 4 | `b C H W` | `b C H W` | 1x1 convs `C->f->C`, then residual add | Linear dgrad/wgrad |
| Project fc1 + GELU + fc2 | 1 | `b C H W` | `b 4 H W` | 1x1 convs `C->C->4` | Linear dgrad/wgrad |
| MSE loss | 1 | Prediction and target | Scalar | `sum((p-t)^2) / (b*c*H*W)` | FP32 gradient wrt prediction |

## Op Costs

Weights include biases. Cached activations are first-order retained-tensor estimates without
activation checkpointing; BF16 pointwise tensors are `b*C*T*2 B = 8.590 GB` per `C`-channel
tensor. FFT FLOPs use `2.5 N log2 N` per real 2D plane; the FFT is memory-bound, so FLOPs are a
poor proxy for its time.

| Op | Count | Parameters (M) | Weights (GB) | Cached activations (GB) | Forward FLOPs (TFLOPs) | Backward FLOPs (TFLOPs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lift fc1 / GELU / fc2 | 1 | 0.000352 | <0.001 | 19.327 | 0.172 | 0.344 |
| Spectral FFT + IFFT | 4 | 0 | 0 | ~0 (linear, nothing saved) | 2.062 (est.) | 2.062 (adjoint) |
| Spectral mode mixing (complex) | 4 | 4.194 (complex) | 0.034 | 0.034 (truncated spectrum) | 0.001 | 0.002 |
| Skip 1x1 | 4 | 0.004 | <0.001 | 34.360 | 0.550 | 1.100 |
| Add + GELU | 4 | 0 | 0 | 34.360 | Elementwise | Elementwise |
| Channel MLP fc1 | 4 | 0.002 | <0.001 | 34.360 | 1.100 | 2.199 |
| Channel MLP GELU | 4 | 0 | 0 | 68.719 | Elementwise | Elementwise |
| Channel MLP fc2 | 4 | 0.002 | <0.001 | 68.719 | 1.100 | 2.199 |
| Project fc1 / GELU / fc2 | 1 | 0.000340 | <0.001 | 25.770 | 0.172 | 0.344 |
| MSE loss | 1 | 0 | 0 | 4.295 | Not counted | Not counted |
| **Total** | | **4.200** | **0.034** | **289.944** | **3.094 (matmul) + 2.062 (FFT)** | **6.188 (matmul) + 2.062 (FFT)** |

Matmul totals include lift, skip, MLP, project and the spectral mode mixing. Backward matmul is
estimated as twice forward; the FFT backward is one adjoint transform, i.e. the same cost as
forward. Elementwise ops, the zero-fill, truncation and reshapes are excluded.

Transient (not saved) memory: the spectral island upcasts to FP32 and the real FFT output is
complex, so one layer alive at once is `b*C*T*4 B + b*C*H*(W/2+1)*8 B = 34.4 GB` before
sharding, plus cuFFT workspace (unknown).

### Width sweep

The activation memory scales linearly in `C` and the pointwise FLOPs quadratically; the weights
are dominated by the spectral tensor, `n*C²*kh*kw*8 B`.

| `embed_dim` | Weights (GB) | Saved activations (GB) | Spectral transient (GB) | Pointwise forward TFLOPs | FFT forward TFLOPs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 (chosen) | 0.034 | 289.9 | 34.4 | 3.09 | 2.06 |
| 32 | 0.134 | 573.4 | 68.7 | 12.23 | 4.12 |
| 64 | 0.537 | 1140.4 | 137.5 | 48.65 | 8.25 |
| 128 | 2.149 | 2274.5 | 274.9 | 194.06 | 16.49 |

The `mp` config initially carried `embed_dim: 128` (an untuned placeholder). It needs 2.27 TB
of saved activations at global batch 16, which no 16-GPU mesh fits without activation
checkpointing. It was changed to 16 so that the plan has a feasible target; 32 is feasible
only with checkpointing or more GPUs.

## Memory Summary

| Category | Estimate (GB) | Dominant dimensions | Notes |
| --- | ---: | --- | --- |
| Weights | 0.034 | `n*C²*kh*kw` complex | 4.194M complex64 + 6,068 FP32 |
| Gradients | 0.034 | Parameter count | |
| Optimizer state | 0.067 | Parameter count | Two Adam tensors, complex for the spectral weights |
| Saved activations | 289.944 | `b`, `C`, `T` | BF16 pointwise tensors; FP32 loss difference |
| **Total per GPU** | **290.079** | Batch and grid | Before sharding; plus a 34.4 GB transient |

This is not peak runtime memory. It excludes FP32 input/target residency (4.295 GB each at
`b=16`), cuFFT workspaces, communication buffers, CUDA context,
DALI prefetching, and allocator overhead.

## Bottleneck

| Bottleneck | Evidence | Candidate sharding dimensions |
| --- | --- | --- |
| Saved activations | 289.944 GB versus 0.134 GB for weights, gradients and Adam state (2,160x) | Batch with DP; grid (H, W) with SP1 and SP2 |
| Memory bandwidth | FFT and pointwise layers have low arithmetic intensity; 5.2 TFLOPs forward in total | Same as above |
| Spectral transients | 34.4 GB per layer alive | Grid with SP1 x SP2 |
| Spectral weights | 0.034 GB, small | Not a capacity problem; shardable over wavenumbers for free with the spatial groups |
| Tensor-parallel candidates | 24 KB of pointwise weights in total | None: TP shards weights, which are negligible |
