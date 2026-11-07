# ConvNeXt DP + SP1 + SP2 Parallelism Plan

This plan uses the fixed model contract in [analysis.md](analysis.md). The
primary target is eight NVIDIA 40 GB GPUs, with four NVLink-connected GPUs per
node, global batch size 4, and a throughput objective.

## Constraints and Candidate Mesh

| Field | Value |
| --- | --- |
| Representative model configuration | 16 layers, `e=1536`, `f=6144`, feature grid `256 x 256` |
| Memory target per device | 40 GB |
| Global batch, microbatch, accumulation | Primary: `4 = 2 x dp(2) x 1` |
| Representative input | `(4, 2, 4096, 4096)` globally |
| Available devices | 8 primary; projections for 4 and 16 |
| Hardware topology | Four GPUs per node with NVLink |
| Bottleneck | Saved activations; MLP compute |
| Candidate mesh | Primary: `dp=2, tp=1, sp1=4, sp2=1` |
| Data-parallel wrapper | DDP with SP1-SP2 wgrad-reduction hook |
| Sequence parallelism | `false`; SP1-SP2 shards the feature grid directly |

SP1 partitions feature-grid height and SP2 partitions width. The primary mesh
keeps four-way height splitting within each node and uses DP across nodes.
Each spatial replica processes two samples. Tensor parallelism is unsupported.

## Operation Sharding and Communication

For the primary mesh, local batch `b_dp=2`, local height `h_s=64`, local
width `w_s=256`, hidden size `e=1536`, and MLP width `f=6144`.

| Op group | Mesh | Logical | Sharded | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- | --- | --- |
| Data / targets | `dp x sp1 x sp2` | `(4,2,4096,4096)` | `(2,2,1024,4096)` | None; DALI reads the rank's tile | None | None |
| Patch stem | Same | `b c H W -> b e h w` | `(2,2,1024,4096) -> (2,1536,64,256)` | None; whole patches stay local | None | Spatial sum, DP average |
| Channel LayerNorm | Same | `b h w e -> b h w e` | `(2,64,256,1536)` | None; channel axis is complete | None | Spatial sum, DP average |
| Depthwise convolution | Same | `b e h w -> b e h w` | `(2,1536,64,256) -> halo (2,1536,70,256) -> core` | Neighbor exchange over SP1; SP2 inactive | Return halo gradients to owners and add to core gradients | Spatial sum, DP average |
| FC1 / GELU / FC2 | Same | `b h w e -> b h w f -> b h w e` | `(2,64,256,1536) -> (2,64,256,6144) -> same core` | None | None | Spatial sum, DP average for weights/biases |
| Layer scale + residual | Same | `R + gamma*U` | Local core | None | None | Spatial sum, DP average for gamma |
| Head + unpatch | Same | `b h w e -> b c H W` | `(2,64,256,1536) -> (2,2,1024,4096)` | None | None | Spatial sum, DP average |
| MSE / metrics | Same | Global spatial statistics | Local sums before reduction | SP1-SP2 all-reduce; DP average for reporting | Identity backward through spatial sum | None |
| Epoch plots | Spatial group | Full fields for visualization | Gather local fields | Width then height all-gather of detached fields | None | None |

The depthwise convolution exchanges three neighboring feature rows on SP1.
Global boundaries use zero halos to preserve serial convolution padding.
Backward returns halo gradients to their owners and sums them with the local
core gradients. For 2D tiles, width exchange precedes height exchange so that
corners are included; backward reverses this order.

The transport uses all-to-all-single with nonzero counts only for adjacent
peers. At the primary mesh, one BF16 edge is 4.719 MB. An interior rank sends
9.437 MB per block in forward and the same amount in backward, or
301.990 MB per step across all 16 blocks. Endpoint ranks send half as much;
receive volume equals send volume.

Channel normalization, MLPs, layer scales, and the patch-aligned stem and head
operate locally. Training outputs stay sharded; detached fields are gathered
only for plotting.

## Parameter Metadata Handoff

Logical operation shapes use mathematical weight order. Parameter and local
shapes use actual PyTorch storage order. Let `q` index batch and feature
positions, `G` denote the output gradient, and `Xpatch` and `Xnbr` denote stem
patches and depthwise neighborhoods.

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `stem.0.weight` | `e i u v` | Same | `(1536,2,16,16)` | Same | `[None,None,None,None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e] Xpatch[q,i,u,v]` |
| `stem.0.bias` | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e]` |
| `stem.1.norm.weight`, `blocks.*.norm.weight`, `norm.weight` | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e] Xhat[q,e]`, normalized input `Xhat` |
| Corresponding LayerNorm biases | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e]` |
| `blocks.*.depthwise.weight` | `e u v` | Same | `(1536,1,7,7)` | Same | `[None,None,None,None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e] Xnbr[q,e,u,v]` |
| `blocks.*.depthwise.bias` | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e]` |
| `blocks.*.mlp.fc1.weight` | `e f` | Same | `(6144,1536)` | Same | `[None,None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q X[q,e] G[q,f]` |
| `blocks.*.mlp.fc1.bias` | `f` | Same | `(6144,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,f]` |
| `blocks.*.mlp.fc2.weight` | `f e` | Same | `(1536,6144)` | Same | `[None,None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q Z[q,f] G[q,e]` |
| `blocks.*.mlp.fc2.bias` | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q G[q,e]` |
| `blocks.*.gamma` | `e` | Same | `(1536,)` | Same | `[None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q U[q,e] G[q,e]` |
| `head.weight` | `e o` | Same | `(512,1536)` | Same | `[None,None]` | `["sp1-sp2"]` | `["sp1-sp2"]` | `sum_q X[q,e] G[q,o]` |

Ordinary DP wgrad synchronization is handled by DDP and is not included in
`reduce`. Every parameter contracts over spatial positions, so its local
gradient is partial and requires an SP1-SP2 sum. The loss uses the full spatial
denominator for each DP batch; gradients are summed over SP and averaged over DP.

All parameters have the same reduction metadata. The existing DDP hook fuses
these reductions into an all-reduce over `dp-sp1-sp2`, divided by `dp=2`.
This group spans all eight GPUs. The FP32 gradient payload is 1.220 GB before
collective protocol traffic. Shared parameters are synchronized at initialization.

## Topology Handoff

| Field | Planned value |
| --- | --- |
| World size | 8 |
| Launch method | `sbatch --nodes=2 submit_pm_mp.sh ...` with four tasks and GPUs per node |
| Mesh dimensions | `dp=2, tp=1, sp1=4, sp2=1` |
| Sequence parallelism | `false` |
| Rank order | `tp-sp1-sp2-dp` |
| Group tokens to build | `dp`, `tp`, `sp1`, `sp2`, `sp1-sp2`, `tp-sp1-sp2`, `dp-sp1-sp2` |
| Hardware topology | One contiguous SP1 x SP2 tile per NVLink node |
| Divisibility | `4096 % 16 == 0`; `256 % 4 == 0`; `4 % 2 == 0`; no padding |

The world-size invariant is:

```text
8 = dp(2) * tp(1) * sp1(4) * sp2(1)
```

With `tp-sp1-sp2-dp` and four consecutive ranks per node:

```text
SP1 groups: (0,1,2,3), (4,5,6,7)
DP groups: (0,4), (1,5), (2,6), (3,7)
TP and SP2 groups: singleton groups
```

Halo communication remains within NVLink nodes. The fused DP-spatial gradient
reduction spans both nodes. The group set includes all atomic dimensions, the
combined spatial plane, the full model-parallel group for initialization, and
the DP-spatial composite for gradient reduction.

## Expected Per-GPU Memory

Weights, gradients, and optimizer states remain replicated. Saved activations
scale with the local batch and spatial area, with an additional term for the
halo-extended depthwise inputs. All values below use decimal GB.

| GPUs | Mesh (`tp=1, sp2=1`) | Weights | Gradients | Optimizer | Saved activations | I/O residency | Communication and temporary | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | `dp=1, sp1=4` | 1.220 | 1.220 | 2.440 | 43.395 | 0.336 | Unknown | 48.611 + unknown |
| 8 | `dp=2, sp1=4` | 1.220 | 1.220 | 2.440 | 21.698 | 0.168 | Unknown | 26.745 + unknown |
| 16 | `dp=4, sp1=4` | 1.220 | 1.220 | 2.440 | 10.849 | 0.084 | Unknown | 15.813 + unknown |

I/O residency includes FP32 input and target tensors plus a BF16 model output.
Halo packing, temporary activation gradients, autocast weight copies, kernel
workspaces, CUDA context, DALI prefetching, NCCL internals, and allocator
fragmentation remain unknown. These are planning estimates, not measured peaks.

## Scale Plan

For every row:

```text
global_batch = microbatch_per_dp_rank * dp * accumulation = 4
```

| Scale | Model/input configuration | Mesh | Microbatch | Reason for changing model-parallel dimensions |
| --- | --- | --- | ---: | --- |
| 4 GPUs | Representative | `dp=1,tp=1,sp1=4,sp2=1` | 4 | Capacity comparison; estimated memory exceeds 40 GB |
| 8 GPUs, primary | Representative | `dp=2,tp=1,sp1=4,sp2=1` | 2 | Hold SP1/SP2 fixed and add DP across two nodes |
| 16 GPUs | Representative | `dp=4,tp=1,sp1=4,sp2=1` | 1 | Hold SP1/SP2 fixed and add DP across four nodes |

## Bottleneck and Rationale

Saved activations dominate the unsharded first-order state. SP1-SP2 shards
feature-grid activations and MLP compute, while only depthwise convolution
requires neighboring spatial values. Keeping `sp1=4, sp2=1` within each
four-GPU node places frequent halo exchanges on NVLink; DP is the scale-out axis.

Height splitting preserves contiguous complete rows in the input layout.
Local width reads select separated row segments and can stall data delivery.
Full-image reads followed by cropping provide an alternative when splitting
width. File chunking and caching affect the cost of either read pattern.

## Pros, Cons, and Risks

Pros:

- SP1-SP2 reduces activation storage and compute throughout the model.
- Only depthwise convolutions require per-block activation communication.
- Halo communication stays within each NVLink node.
- Serial and spatial models use compatible parameter shapes and state dictionaries.

Cons:

- Model parameters and optimizer states remain replicated.
- Every block adds forward and backward halo exchange.
- All parameter gradients require spatial reduction as well as DP synchronization.
- Global batch 4 limits DP scaling to four replicas.

Risks:

- Missing halo-gradient contributions or SP1-SP2 wgrad reductions produce incorrect training.
- Two-dimensional tiles require corner exchange and correct backward accumulation.
- Local width reads can become an I/O bottleneck.
- Runtime memory can exceed static estimates due to DALI, kernels, NCCL, and allocator behavior.

## Affected Files and Model Boundaries

- `networks/convnext.py`: serial patch stem, ConvNeXt blocks, and reconstruction head.
- `networks/convnext_mp.py`: spatial depthwise convolution and parameter metadata.
- `distributed/spatial_sharding.py`: patch-aligned tile construction and validation.
- `distributed/helpers.py`: neighbor exchange and contiguous edge packing.
- `distributed/mappings.py`: forward halo exchange and backward gradient accumulation.
- `train.py`, `train_mp.py`: model selection using the existing training contract.
- `config/ConvNeXt.yaml`: `small` and `big` configurations.
- Tests: halo and numerical-equivalence coverage.

## Validation

1. Verify rank memberships and node placement for the primary and scale-out meshes.
2. Compare serial and spatial forward outputs for height stripes, width stripes,
   and 2D tiles, including uneven tiles and global boundaries.
3. Compare reconstructed input gradients against the serial baseline.
4. Compare all parameter gradients after SP1-SP2 and DP reductions.
5. Verify replicated parameters remain synchronized across repeated optimizer updates.
6. Check eager and compiled execution, including BF16 and thin halo buffers.
7. Measure peak allocated/reserved CUDA memory and NVML high-water marks.
8. Benchmark samples/second and data waits with fixed batch, dataset size,
   precision, and launch settings.

## Alternatives Considered

- **DP4 x SP2:** Uses one sample per spatial replica and two GPUs per image;
  also fits on eight GPUs at global batch 4.
- **Width stripes or 2D tiles:** Supported by halo exchange; full-image reads
  followed by cropping can avoid fragmented local reads.
- **Pure DP:** Reduces local batch but does not partition activations within an image.
- **FSDP:** Targets model state rather than the dominant activation memory and
  requires compatible spatial-gradient synchronization.
- **Activation checkpointing:** Reduces retained activations at the cost of
  recomputation, including repeated halo exchanges.
