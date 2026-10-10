# FNO DP + SP1 x SP2 Parallelism Plan

This plan follows [the ViT plan](../vit/parallelism-plan.md) and uses the fixed model contract
in [analysis.md](analysis.md). The target is 16 NVIDIA 40 GB GPUs, four NVLink-connected GPUs
per node, global batch size 16, and a throughput objective. It was written by applying the ViT
plan's method unchanged wherever possible; [comparison.md](comparison.md) lists where the
method had to be extended and how the result compares to `networks/fno_mp.py`.

## Constraints and Candidate Mesh

| Field | Value |
| --- | --- |
| Representative model configuration | 4 blocks, `C=16`, `kh=kw=64`, `T=4096x4096` |
| Memory target per device | 40 GB |
| Global batch, microbatch, accumulation | Primary: `16 = 4 x dp(4) x 1` |
| Representative input | `(16, 4, 4096, 4096)` globally |
| Available devices | 16 primary; projections for 4 and 8 |
| Hardware topology | Four GPUs per node with NVLink |
| Bottleneck | Saved activations and memory bandwidth; weights are negligible |
| Candidate mesh | Primary: `dp=4, tp=1, sp1=2, sp2=2` |
| Data-parallel wrapper | DDP with the non-DP reduction hook (SP1-SP2 for the pointwise layers) |
| Sequence parallelism | Not applicable: there is no sequence axis, the grid itself is sharded |
| Interface | Spatially sharded: each rank loads, takes and returns its `(H/sp1, W/sp2)` shard; the loss reduces over SP1-SP2 |

Pure DP (`dp=16`, one image per GPU) fits 40 GB on its own (21.1 GB estimated), needs no
activation communication and is likely the fastest. Spatial parallelism is the requested
feature and is also the only way to go beyond global batch 16 GPUs, so the minimum active
tile is `sp1=2, sp2=2` (4 GPUs per replica), with the remaining GPUs assigned to DP. Unlike the
ViT, `sp1` and `sp2` are **independent axes**: `sp1` shards H, `sp2` shards W, and the FFT is
a pencil transform over them. `tp=1`: tensor parallelism has no weights to shard here and
would cost more than it saves (see Alternatives).

## Operation Sharding and Communication

Let `sp=sp1*sp2`. For the primary mesh: local batch `b_dp=4`, local grid
`(H/sp1, W/sp2) = (2048, 2048)`, `C=16` unsharded. `T1..T8` are the all-to-all transposes
inside one spectral layer; "c<->w" means the channel dim and the w dim swap which one is
sharded.

| Op group | Mesh | Logical | Sharded | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- | --- | --- |
| Data loading (input and target) | `dp x sp1 x sp2` | `b 4 H W` | `b/dp 4 H/sp1 W/sp2` | Each rank reads only its (H, W) tile of the input and the target (`spatial_read_mode: local`, an HDF5 hyperslab), from the `spatial_shard` of PR #3; no collective | None | None |
| Lift (pointwise) | `dp x sp1 x sp2` | `b 4 H W -> b C H W` | spatial shards, channels local | None | None | All-reduce SP1-SP2; DDP over DP |
| Spectral T1: c<->w | `sp2` | `b C H W` | `b C H/sp1 W/sp2 -> b C/sp2 H/sp1 W` | All-to-all SP2 (`(sp2-1)/sp2` of the local FP32 field) | All-to-all SP2 (inverse) | None |
| `rfft` w, truncate to `kw` | local | `b C H W -> b C H kw` | `b C/sp2 H/sp1 kw` | None | None | None |
| Spectral T2: w<->c | `sp2` | | `-> b C H/sp1 kw/sp2` | All-to-all SP2 (truncated, small) | All-to-all SP2 | None |
| Spectral T3: c<->h | `sp1` | | `-> b C/sp1 H kw/sp2` | All-to-all SP1 (truncated, small) | All-to-all SP1 | None |
| `fft` h, truncate to `kh` | local | | `b C/sp1 kh kw/sp2` | None | None | None |
| Spectral T4: h<->c | `sp1` | | `-> b C kh/sp1 kw/sp2` | All-to-all SP1 (tiny) | All-to-all SP1 | None |
| Mode mixing (complex) | `sp1 x sp2` | `b C kh kw, C C kh kw -> b C kh kw` | `b C kh/sp1 kw/sp2, C C kh/sp1 kw/sp2` | None | None | **None** (see metadata) |
| T5..T8 (inverse) | `sp1`, `sp2` | | reverse of T4..T1, with `ifft` over h and zero-fill (`kh->H`), zeroing of the self-conjugate columns, `irfft` over w | 4 all-to-alls; only T8 moves the large FP32 field, T5..T7 act on the truncated spectrum | 4 all-to-alls | None |
| Skip 1x1 + add + GELU | `dp x sp1 x sp2` | `b C H W` | spatial shards | None | None | All-reduce SP1-SP2; DDP over DP |
| Channel MLP | `dp x sp1 x sp2` | `b C H W` | spatial shards | None | None | All-reduce SP1-SP2; DDP over DP |
| Project (pointwise) | `dp x sp1 x sp2` | `b C H W -> b 4 H W` | spatial shards | None | None | All-reduce SP1-SP2; DDP over DP |
| Loss and metrics | `dp x sp1 x sp2` | `b 4 H W` | Prediction and target stay `b/dp 4 H/sp1 W/sp2`; the sum of squares is formed on the shard | All-reduce SP1-SP2 of 1 scalar (`mse_loss`) and 1 per sample and channel (`rmse_spatial`) | Identity backward: each rank gets the gradient for its own shard | DDP only |

Communication volume per GPU at the primary mesh (analytical): each spectral layer forward moves
about `2 x 0.537 = 1.07 GB` in T1 and T8 (the real FP32 field, `1.07 GB` locally, half of it
sent to the one SP2 peer) and about `0.03 GB` in the truncated transposes; backward is the same.
Four layers: **8.9 GB sent per step per GPU**, almost all over the SP2 group. The interface needs
no gather: the model returns its shard and the loss all-reduces a few scalars per sample. Temporary send, receive and concatenation buffers are about three times the local
FP32 field, `3.2 GB`.

The pointwise layers need no activation communication under spatial sharding, because they act
per grid point. The spectral layer is the only op that couples grid points, and truncation to
`kh x kw` modes makes every transpose after T1 two to three orders of magnitude smaller than T1
and T8.

## Parameter Metadata Handoff

Logical operation shapes use mathematical weight order. Parameter and local shapes use actual
PyTorch storage order. Spectral weights are sharded over the *spatial* groups, along their
wavenumber dims.

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `blocks.*.spectral.weight` | `C C kh kw` (complex) | `C C kh/sp1 kw/sp2` | `(16,16,64,64)` c64 | `(16,16,32,32)` | `[null,null,"sp1","sp2"]` | `["tp"]` | `[]` | Mode shards are owned by one spatial rank each; the all-to-all backward routes every contribution to the owner, so the local wgrad is complete |
| `lift.fc1.weight` | `4 -> C` | Same (tp=1) | `(16,4)` | Same | `["tp",null]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Replicated over spatial ranks that see different grid points |
| `lift.fc1.bias` | `C` | `C/tp` | `(16)` | Same | `["tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Same |
| `lift.fc2.weight` | `C -> C` | `C/tp C` | `(16,16)` | Same | `[null,"tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Same |
| `lift.fc2.bias` | `C` | Same | `(16)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Row-parallel bias is replicated over TP |
| `blocks.*.skip.{weight,bias}` | `C C` / `C` | Same | `(16,16)` / `(16)` | Same | `[null,null]` / `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Replicated pointwise layer on spatial shards |
| `blocks.*.mlp.fc1.{weight,bias}` | `C f` | `C f/tp` | `(32,16)` / `(32)` | Same (tp=1) | `["tp",null]` / `["tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Output-feature TP shard (unused at tp=1) |
| `blocks.*.mlp.fc2.weight` | `f C` | `f/tp C` | `(16,32)` | Same (tp=1) | `[null,"tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Input-feature TP shard |
| `blocks.*.mlp.fc2.bias` | `C` | Same | `(16)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Row-parallel bias is replicated over TP |
| `project.*` | as `lift.*` | | | | | | | Same rules as `lift.*` |

General rule (a restatement of the ViT plan's, in terms that cover both models): a parameter
needs a reduction over a group `G` if and only if it is replicated over `G` and the
activations its wgrad contracts over are sharded over `G`. The spectral weights are not
replicated over the spatial groups (they are sharded over them), so they need none. Spectral
weights are replicated over `tp`, where the activations are replicated too, so their
gradients are identical and `tp` appears only in `shared`.

Ordinary DP synchronization is handled by DDP and is not in `reduce`. Because the spectral
weights (signature `()`) and the pointwise weights (signature `("sp1-sp2",)`) differ, the single-
signature fused reduction hook does not apply and the general per-bucket hook is used.

## Initialization Handoff

The distributed model must start from an initialization that is **statistically equivalent** to
the serial model's, for every parameter, independent of the mesh. It does not have to be
bitwise identical: each rank draws its own shard, so the global tensor depends on the mesh, but
the distribution must not. This is a plan item of its own, because the model-parallel layers
are often reused from another model with the constants of that model.

Rules:

1. **Compute every scale from global sizes**, passed to the layer explicitly. Never derive it
   from `weight.shape`: a row-parallel weight has local shape `(out, in/tp)`, so a local
   fan-in is wrong by `tp`, and the same holds for any fan-out based scheme on a column-parallel
   weight. Constants that depend on no size at all (such as `0.02`) must be justified at the
   model's width.
2. **Match biases as well as weights.** Biases change the activation scale at initialization
   and the early training speed (measured below).
3. **Sharded parameters:** independent draws per shard from the same distribution, no
   synchronization. **Replicated parameters** (anything with a non-empty `shared`): identical on
   every rank of the group, via `init_params_for_shared_weights`.
4. **Complex parameters:** state the variance of the complex variable, `E|w|^2`. `torch.randn`
   with a complex dtype has `E|w|^2 = 1`, split evenly over the real and imaginary parts.

| Parameter pattern | Serial init | Distributed rule (global sizes) | Current distributed implementation |
| --- | --- | --- | --- |
| `blocks.*.spectral.weight` | complex normal, `E\|w\|^2 = gain / C_in` | the same, `C_in` the global input channels; independent per shard | Matches |
| pointwise weights (`lift`, `skip`, `mlp`, `project`) | PyTorch conv default: std `1/sqrt(3 * fan_in)` | normal, std `1/sqrt(3 * fan_in_global)`; `fan_in_global` is the unsharded input width (`C` for fc1, `f` for fc2) | Matches (was `trunc_normal_(std=0.02)`, a constant inherited from the ViT): `init_pointwise_` in `networks/fno.py`, applied to the ViT's tensor-parallel layers from the FNO model, with `fan_in = in_features_local * tp` for row-parallel layers |
| spectral weights, k-dependent decay (`spectral_init_k0`) | the global grid of `spectral_wavenumbers`, times `1/(1+(k/k0)^2)` | the slice of that grid that the shard owns, from the **global** mode indices (`h0 = sum(kh_shapes[:rank_h])`, likewise `w0`), never from the local shape | Matches; `test_k_dependent_init_uses_the_global_wavenumber` compares each shard's per-mode power with the decay of its global wavenumber on uneven meshes, and fails if the local index is used |
| pointwise biases | uniform `+-1/sqrt(fan_in)` | the same with `fan_in_global`; row-parallel and replicated biases identical across the group | Matches (was zeros) |

The k-dependent decay of the spectral weights (`spectral_init_k0`, default 8 in `config/FNO.yaml`) follows
the same rule for any index-dependent quantity: it must be a function of the *global* mode index. A distributed
weight is a shard of the global `(kh, kw)` grid, and along h the layout is the positive frequencies first and the
negative ones after, so the wavenumber of a local index depends on the rank's offset. Measured on the smoke
problem (serial FNO, 3 seeds, 2400 iterations, fp32): the final validation loss was 4.5e-5 without the decay and
3.2e-5 with `k0=8` (3.1e-5 with `k0=4`, 3.9e-5 with `k0=16`), better in all three seeds. Fields usually have little
energy at high wavenumbers, so those weights get tiny gradients, hardly move from their random values and add noise.
The smoke data is exactly band-limited, the extreme case, so the size of the effect on real fields is not known.

Measured effect of the original distributed init, before it was fixed, at the representative width
(`C=16`, 4 blocks, `modes=24`, one rank, 480 iterations of the smoke problem at `lr=5e-4`; one seed,
so indicative only):

| Variant | Output std at init | Train loss at 120 / 240 / 360 / 480 |
| --- | ---: | --- |
| Serial, default init | 0.110 | 0.645 / 0.037 / 0.016 / 0.014 |
| Distributed, as implemented (std 0.02, zero bias) | 0.0000 | 0.999 / 0.780 / 0.631 / 0.596 |
| Distributed, fan-in weights, zero bias | 0.005 | 0.714 / 0.081 / 0.033 / 0.032 |
| Distributed, fan-in weights + random bias | 0.110 | 0.690 / 0.042 / 0.017 / 0.015 |
| Serial, biases zeroed | 0.003 | 0.695 / 0.075 / 0.027 / 0.023 |

At this width the constant `0.02` is 5 to 14 times smaller than the fan-in value, so the signal
shrinks by that factor per linear layer, and at initialization the output std and the
gradients are about 1e-13 to 1e-16. The weight scale is the dominant effect; the biases add a
second factor of about two. A `std=0.02` is appropriate at the ViT's width of 1024
(`1/sqrt(1024) = 0.03`), which is why it does not show up there.

The existing equivalence tests copy the serial weights into the shards, so they cannot see an
initialization difference; this item needs its own test (see Validation). The literature on FNO initialization is a further step beyond statistical
equivalence with the serial model: it studies which scale is *critical* for signal
propagation with mode truncation (arXiv 2310.06379 and 2507.21833); the serial reference
itself should follow it once it is settled, and the distributed model then inherits it through
the rules above.

## Topology Handoff

| Field | Planned value |
| --- | --- |
| World size | 16 |
| Launch method | `srun -N 4 --ntasks-per-node=4 --gpus-per-node=4 ... train_mp.py --yaml_config config/FNO.yaml --config mp --sp1 2 --sp2 2` |
| Mesh dimensions | `dp=4, tp=1, sp1=2, sp2=2` |
| Rank order | `tp-sp1-sp2-dp` |
| Group tokens to build | `dp`, `tp`, `sp1`, `sp2`, `sp1-sp2`, `tp-sp1-sp2`, `dp-sp1-sp2` (all already built by `comm.init_model_parallel_info`) |
| Hardware topology | One contiguous `sp1 x sp2` tile per NVLink node |
| Divisibility | `H, W >= sp1, sp2`; `kh >= sp1`, `kw >= sp2`; `C_in, C_out >= sp1, sp2`; uneven splits are supported (balanced `compute_split_shapes`), no padding |

```text
16 = dp(4) * tp(1) * sp1(2) * sp2(2)
```

With `tp-sp1-sp2-dp` and `tp=1`, `sp1` ranks vary fastest. Node 0 holds ranks 0-3, which is
one full spatial tile:

```text
SP1 groups: (0,1), (2,3)
SP2 groups: (0,2), (1,3)
SP1-SP2 group: (0,1,2,3)
example DP group: (0,4,8,12)
```

All spectral all-to-alls and the output gather stay on NVLink; only DP gradient
synchronization (small: the pointwise gradients and the sharded spectral weights) crosses nodes.

## Expected Per-GPU Memory

**Measured correction (read this first).** The first 16-GPU job with global batch 16
(`dp=1, sp=4x4`) ran out of memory on 40 GB GPUs, so the table below is too optimistic by about
a third. Two measurements explain it. (1) A single rank running the distributed model on one GPU's
tile, `(16, 16, 1024, 1024)` under bf16 autocast, keeps **21.8 GB** after the forward pass, against
18.1 GB analytically; the first-order accounting misses some saved tensors. (2) Before a fix, the
row-parallel bias promoted the residual stream to fp32 (23.9 GB kept). On top of that come about 3 GB of
all-to-all buffers and, in the failing job, about 6 GB that the PyTorch allocator does not
count (NCCL communicators, cuFFT workspaces, the CUDA context): 38.5 GB in use against 31.5 GB
allocated. So a realistic budget at this tile is about 30 to 35 GB, and **global batch 8** is the
fitting choice on 16 GPUs (about 20 GB); all `dp x sp` splits of 16 GPUs have the same per-GPU memory.
The static estimates exclude non-PyTorch memory, which the ViT plan already warns about, but
the FNO's all-to-alls make it large. Activation checkpointing remains the lever for larger batches;
it is not implemented.

Estimates from `analysis.md`. With sharded I/O everything scales with `dp*sp`: the saved
activations, including the loss difference, and the temporary memory, which is the FP32 input and
target shards, the BF16 output shard and the spectral transient. Communication is three times the
local FP32 field.

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `dp=1,sp=2x2` | 0.008 GB | 0.008 GB | 0.017 GB | 72.49 GB | 11.28 GB | 12.88 GB | **96.7 GB: does not fit** |
| 8 | `dp=2,sp=2x2` | 0.008 GB | 0.008 GB | 0.017 GB | 36.24 GB | 5.64 GB | 6.44 GB | **48.4 GB: does not fit** |
| 16 | `dp=4,sp=2x2` | 0.008 GB | 0.008 GB | 0.017 GB | 18.12 GB | 2.82 GB | 3.22 GB | **24.2 GB + unknown** |
| 16 | `dp=16` (reference) | 0.034 GB | 0.034 GB | 0.067 GB | 18.12 GB | 2.82 GB | 0 | 21.1 GB + unknown |

At 16 GPUs every `dp x sp` split with `sp > 1` comes to about the same 24.2 GB per GPU, because
the sharded interface divides everything by the GPU count; the splits differ in communication
(8.9 GB sent per step for `sp=2x2`, 13.2 to 13.3 GB for `sp=2x4` and `4x4`) and in the batch per
replica. The 0.3 GB communication allowance of the ViT plan does not carry over: the transposes
move FP32 fields, so the buffers are `3 x 1.07 GB`. cuFFT workspaces, CUDA context, DALI
prefetching and allocator fragmentation remain unknown.

## Scale Plan

For every row `global_batch = microbatch_per_dp_rank * dp * accumulation = 16`.

| Scale | Model/input configuration | Mesh | Microbatch | Reason for changing model-parallel dimensions |
| --- | --- | --- | ---: | --- |
| 4 GPUs | Representative | `dp=1,sp=2x2` | 16 | Smallest active spatial tile; **exceeds 40 GB, needs activation checkpointing or a smaller model** |
| 8 GPUs | Representative | `dp=2,sp=2x2` | 8 | Hold the tile fixed and add DP; **exceeds 40 GB** |
| 16 GPUs, primary | Representative | `dp=4,sp=2x2` | 4 | First feasible scale at 4096² |

## Bottleneck and Rationale

Saved activations dominate by three orders of magnitude, and they scale with both batch and
grid, so batch (DP) and grid (SP1 x SP2) are the only effective axes. Spatial sharding
costs an all-to-all of the FP32 field per spectral layer in each direction, but the
truncation to `kh x kw` modes keeps every other transpose cheap. Keeping the `sp1 x sp2`
tile inside one NVLink node puts that traffic on the fastest links.

## Pros, Cons, and Risks

Pros:

- Pointwise layers shard perfectly with no communication; spectral weights shard for free.
- All large-volume collectives are confined to the SP2 group inside a node.
- The same tile scales from 4 to 16 GPUs through DP.

Cons:

- Only 16 GPUs fit the representative model; below that, activation checkpointing is required.
- Every spectral layer pays about 1.1 GB per GPU per direction on NVLink at this mesh.
- The data loader has to be spatially aware (each rank reads a hyperslab of every image), and
  the loss and metrics need a small reduction; a plot needs a gather of one sample's channel.
- The channel count must be at least `sp1` and `sp2`.

Risks:

- A missing SP1-SP2 reduction on a pointwise weight silently produces wrong training; the
  spectral weights are the opposite case, a spurious reduction would double count.
- Fused Adam rejects the complex spectral weights; the optimizer must be non-fused.
- Initialization constants inherited from another model's layers (the ViT's `0.02`) vanish the signal at small width; fixed, and guarded by `tests/test_fno_init.py`; see Initialization Handoff.
- Large FP32 FFT workspaces and the FP32 spectral island may push the peak above the
  estimate.
- The distributed model cannot be compiled with `fullgraph=True` because the collectives are
  excluded from the graph. The spectral layers are excluded too (`@torch.compiler.disable`): FFT
  fallback kernels broke inductor's stride assertion on the NERSC container's torch at 4096^2.

## Affected Files and Model Boundaries

- `networks/fno_mp.py`: spectral layer, pointwise layers, model, metadata.
- `distributed/fft.py`, `distributed/helpers.py`, `distributed/mappings.py`: pencil FFT, `_transpose`, `distributed_transpose`.
- `networks/vit_mp.py`: shared `MLP` layers with a `layout` argument.
- `train_mp.py`: network selection, the sharded loss and metrics, the plot gather and the non-fused optimizer choice.
- `distributed/spatial_sharding.py`: `make_grid_spatial_shard`, the FNO's shard (a plain (H, W) split), next to the ViT's patch-aligned one. The loaders, `mse_loss`, `rmse_spatial` and the plot gather are PR #3's and are unchanged for the FNO.
- `utils/data_loader.py`, `utils/data_loader_dali.py`, `utils/dali_es_helper.py`, `utils/loss.py`, `utils/metrics.py`: PR #3 (this branch is based on it).
- Tests: serial and distributed equivalence, DDP reduction.

## Validation

1. Compare the serial FNO against the distributed FNO forward output for every mesh in the scale plan, with uneven sizes.
2. Compare every local weight-gradient shard after the SP1-SP2 and DP reductions, with the real DDP hook.
3. Verify that the replicated parameters stay synchronized (`init_params_for_shared_weights`).
4. Verify the self-conjugate (DC and Nyquist) handling against a CPU `irfft2` reference, on GPU.
5. Measure peak allocated/reserved CUDA memory and NVML high-water marks, including cuFFT workspaces.
6. Benchmark samples/second against pure DP; do not claim a throughput improvement without measurement.
7. Check the sharded interface: the loader shards of all spatial ranks tile the image and equal the full loader's slices (both loaders), `mse_loss` and `rmse_spatial` on the shards equal the plain MSE and RMSE of the full fields, and the gradient with respect to each shard equals the corresponding slice of the full gradient, on every mesh in the scale plan.
8. Check initialization equivalence, separately from the weight-copying checks above: build the serial and distributed models, compare the per-parameter weight and bias statistics (gathered over the shards) and the per-layer activation and gradient scale at initialization, on every mesh in the scale plan, and compare the loss curves over a few hundred iterations on a small problem. Statistical agreement, not bitwise. `tests/test_fno_init.py` does this: per-parameter statistics within about 5 sigma of the sampling error of their size (so tiny parameters are only bounds), replicated parameters identical over their `shared` groups, and, over 15 draws each, the median output std and `lift.fc1` gradient norm of the distributed model within a factor of 2 of the serial one. Single networks cannot be compared: at this depth the output std of one draw varies by more than 10x between seeds, in both models.

## Alternatives Considered

- **Pure DP:** fits (21.1 GB), has no activation communication and is likely faster. Rejected
  only because spatial parallelism was requested, and because it does not scale beyond
  global batch 16 GPUs.
- **Tensor parallelism:** the pointwise layers have 24 KB of weights in total, so TP saves no
  memory, yet each TP-sharded MLP block needs an all-reduce of a full activation
  (`0.537 GB` at the primary mesh, 6.4 GB per step). The spectral layer needs all channels
  local, so it would not shard either.
- **Larger `sp1`/`sp2`:** adds transposes and, at fixed batch, lowers `dp`. `dp=1,sp=4x4` and
  `dp=2,sp=2x4` have the same estimated memory (24.2 GB) but more communication (13.3 and 13.2 GB
  sent per step against 8.9 GB), so there is no reason to prefer them at this size.
- **Activation checkpointing:** not modeled quantitatively; it is the lever that makes 4 and 8
  GPUs feasible and larger widths possible, but it is not implemented in the repo.
- **FSDP:** the first-order path is invalid for the same reason as in the ViT plan: the
  pointwise weights have non-DP wgrad reductions.
