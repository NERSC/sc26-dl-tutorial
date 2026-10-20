# Homogeneous Swin DP + TP + SP1 + SP2 Parallelism Plan

This plan uses the fixed model contract in [analysis.md](analysis.md). The
primary target is 16 NVIDIA A100 40 GB GPUs, four GPUs per node, global batch
size 16, and a throughput objective.

## Constraints and Candidate Mesh

| Field | Value |
| --- | --- |
| Representative model configuration | 12 homogeneous layers, `e=1024`, `nh=16`, `f=4096`, `64x64` patch grid, `16x16` windows |
| Memory target per device | 40 GB |
| Global batch, microbatch, accumulation | Primary: `16 = 8 x dp(2) x 1` |
| Representative input | `(16, 4, 4096, 4096)` globally |
| Available devices | 16 primary; scale-out projections for 8, 32, and 64 |
| Hardware topology | Four A100 GPUs per node |
| Bottleneck | Saved activations; MLP and patch/head compute; shifted-window boundaries |
| Candidate mesh | Primary: `dp=2, tp=2, sp1=2, sp2=2` |
| Data-parallel wrapper | DDP with non-DP wgrad-reduction hooks |
| Sequence parallelism | `false`; SP1 and SP2 directly shard the two spatial axes |

Pure DP is simpler and probably faster for the fixed model because it already
fits when batch is divided over 16 GPUs. TP, SP1, and SP2 are active because
all four dimensions were explicitly requested. The eight-rank model-parallel
tile is therefore a functionality target rather than the minimum capacity
requirement; additional devices scale through DP.

For the primary mesh, each DP rank owns a microbatch of 8, each spatial rank
owns a `2048x2048` pixel region or `32x32` patch grid, and each TP rank owns
8 attention heads and MLP width 2048. A local patch grid contains four
`16x16` windows.

## Operation Sharding and Communication

| Op group | Mesh | Logical | Sharded | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- | --- | --- |
| Input/patch convolution | `dp x tp x sp1 x sp2` | `b c H W -> b hp wp e` | `b/dp c H/sp1 W/sp2 -> b/dp hp/sp1 wp/sp2 e` | Loader supplies local crop | None beyond local convolution | DDP; all-reduce SP1-SP2 |
| Position add | same | `b hp wp e` | `b/dp hp/sp1 wp/sp2 e` | None | None | DDP only; parameter itself is spatially sharded |
| LayerNorm | same | `b hp wp e` | `b/dp hp/sp1 wp/sp2 e` | None | None | DDP; all-reduce SP1-SP2 for affine parameters |
| Odd-block cyclic shift | same | roll `hp,wp` by `(-8,-8)` | Roll local `32x32` patch grid | Ring edge exchange with `all_to_all_single` on SP1, then SP2 | Reverse distributed rolls | None |
| Window partition | same | `b hp wp e -> (b*nw) a e` | Four local windows per rank | None | None | None |
| Q/K/V, column TP | same | `... e, e e -> ... e` | `... e, e e/tp -> ... e/tp` | None | All-reduce TP | DDP; all-reduce SP1-SP2 |
| Window attention | same | `(b*nw) nh a d` | local batch/windows, `nh/tp` heads | None; windows are complete after roll | None beyond roll and TP mappings | None |
| Attention projection, row TP | same | `... e, e e -> ... e` | `... e/tp, e/tp e -> ... e` | All-reduce TP | None | DDP; all-reduce SP1-SP2 |
| MLP FC1, column TP | same | `b l e, e f -> b l f` | `b/dp l/(sp1*sp2) e, e f/tp -> ... f/tp` | None | All-reduce TP | DDP; all-reduce SP1-SP2 |
| MLP FC2, row TP | same | `b l f, f e -> b l e` | `... f/tp, f/tp e -> ... e` | All-reduce TP | None | DDP; all-reduce SP1-SP2 |
| Final LayerNorm | same | `b hp wp e` | `b/dp hp/sp1 wp/sp2 e` | None | None | DDP; all-reduce SP1-SP2 |
| Reconstruction head | same | `b hp wp e, e o -> b c H W` | local batch and spatial grid; head replicated | None | None | DDP; all-reduce SP1-SP2 |
| Loss/metrics | same | local squared-error sum | local pixels | All-reduce SP1-SP2; metrics then reduce DP | Identity backward through SP reduction | None |

Odd blocks perform a distributed pre-shift and post-shift on both spatial
axes. A positive roll sends a trailing edge to the right neighbor and receives
from the left; a negative roll sends a leading edge left and receives from the
right. Backward applies the inverse roll. No full spatial all-gather occurs in
a Swin block.

The attention mask is the standard two-dimensional shifted-window mask. It is
constructed from global patch coordinates, sliced to each rank's window-aligned
patch region, and masks cyclic wraparound on both height and width.

## Parameter Metadata Handoff

Logical operation shapes use mathematical weight order. Parameter and local
shapes use actual PyTorch storage order. Ordinary DP synchronization is
provided by DDP and is not listed in `reduce`.

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_embed.proj.weight` | Conv kernel | Same | `(1024,4,64,64)` | Same | `[null,null,null,null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Replicated kernel contracts local batch and local patch-grid positions |
| `patch_embed.proj.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Replicated bias sees local spatial positions |
| `pos_embed` | `hp wp e` | `hp/sp1 wp/sp2 e` | `(1,64,64,1024)` | `(1,32,32,1024)` | `[null,"sp1","sp2",null]` | `["tp"]` | `[]` | Each SP rank owns distinct position rows/columns |
| `blocks.*.norm*.weight` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Affine wgrad contracts local batch and spatial positions |
| `blocks.*.norm*.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Affine wgrad contracts local batch and spatial positions |
| `blocks.*.attn.{q,k,v}.weight` | `e e` | `e e/tp` | `(1024,1024)` | `(512,1024)` | `["tp",null]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Stored output-feature TP shard; spatial positions are a sharded contraction |
| `blocks.*.attn.{q,k,v}.bias` | `e` | `e/tp` | `(1024)` | `(512)` | `["tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Output-feature TP shard; bias sums local positions |
| `blocks.*.attn.proj.weight` | `e e` | `e/tp e` | `(1024,1024)` | `(1024,512)` | `[null,"tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Stored input-feature TP shard; spatial positions are a sharded contraction |
| `blocks.*.attn.proj.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Added once after TP all-reduce; spatial contraction remains partial |
| `blocks.*.mlp.fc1.weight` | `e f` | `e f/tp` | `(4096,1024)` | `(2048,1024)` | `["tp",null]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Stored output-feature TP shard |
| `blocks.*.mlp.fc1.bias` | `f` | `f/tp` | `(4096)` | `(2048)` | `["tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Output-feature TP shard |
| `blocks.*.mlp.fc2.weight` | `f e` | `f/tp e` | `(1024,4096)` | `(1024,2048)` | `[null,"tp"]` | `["sp1-sp2"]` | `["sp1-sp2"]` | Stored input-feature TP shard |
| `blocks.*.mlp.fc2.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Added once after TP all-reduce |
| `norm.{weight,bias}` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Final norm sees local spatial positions |
| `head.weight` | `e o` | Same | `(16384,1024)` | Same | `[null,null]` | `["tp-sp1-sp2"]` | `["sp1-sp2"]` | Replicated head contracts local spatial positions |

The head remains unsharded to stay identical to the minimalist ViT boundary.
This repeats head storage and compute over TP but avoids an additional output
layout transition.

## Weight-Gradient Derivation

Let `r=(b,hp,wp)` denote the axes distributed over DP, SP1, and SP2.

- Linear weights use `dW[e,f] = sum_r X[r,e] * dY[r,f]`. DP handles the
  sharded `b` contraction; `sp1-sp2` is in `reduce` because `hp,wp` are local.
  TP-sharded output features map to stored dimension 0; TP-sharded input
  features map to stored dimension 1. TP does not enter `reduce` because those
  ranks own distinct parameter shards.
- Linear biases use `db[f] = sum_r dY[r,f]` and therefore have the same
  SP1-SP2 reduction. Row-parallel biases are shared over TP and are added only
  after the TP forward all-reduce.
- LayerNorm affine weights use
  `dgamma[e] = sum_r dY[r,e] * xhat[r,e]`; biases use
  `dbeta[e] = sum_r dY[r,e]`. Both require SP1-SP2 reduction.
- Patch weights use
  `dW[o,i,p,q] = sum_{b,hp,wp} X[b,i,hp+p,wp+q] * dY[b,o,hp,wp]`.
  Patch-aligned, nonoverlapping local inputs make the spatial sum partial, so
  the replicated kernel reduces over SP1-SP2.
- Position weights have `dP[hp,wp,e] = sum_b dY[b,hp,wp,e]`. Their spatial
  indices are parameter-sharded rather than contracted, so only DDP reduces
  their replicated batch copies.
- The head uses
  `dW[o,e] = sum_{b,hp,wp} dY[b,hp,wp,o] * X[b,hp,wp,e]`, requiring
  SP1-SP2 reduction for its replicated local contributions.

These non-DP reductions make the first-order FSDP path invalid without a
version-specific communication hook; the implementation will use DDP.

## Topology Handoff

| Field | Planned value |
| --- | --- |
| World size | 16 |
| Launch method | One `torchrun` agent per SLURM node, four local workers per agent |
| Mesh dimensions | `dp=2, tp=2, sp1=2, sp2=2` |
| Mesh invariant | `16 = dp(2) * tp(2) * sp1(2) * sp2(2)` |
| Sequence parallelism | `false` |
| Rank order | `tp-sp1-sp2-dp` |
| Group tokens to build | `dp`, `tp`, `sp1`, `sp2`, `sp1-sp2`, `tp-sp1-sp2`, `dp-tp`, `dp-sp1-sp2`, `dp-tp-sp1-sp2` |
| Hardware topology | Four A100 GPUs per node; one eight-rank model-parallel tile spans two nodes |
| Divisibility | `1024 % tp == 0`, `16 heads % tp == 0`, patch grid `64x64` divides over SP1/SP2, every local `32x32` grid divides by window `16x16`; no padding |

The complete group-token list includes every atomic dimension, every token in
parameter metadata, and the DP-composite candidates for shared/reduced model
parallel groups. It does not create a separate sequence-parallel dimension.

With `tp-sp1-sp2-dp`, ranks 0-7 form the first model-parallel tile:

```text
node 0: ranks 0-3, all TP and SP1 partners for sp2=0
node 1: ranks 4-7, all TP and SP1 partners for sp2=1
node 2: ranks 8-11, the DP peer tile's sp2=0 half
node 3: ranks 12-15, the DP peer tile's sp2=1 half
```

TP and SP1 traffic remains within a four-GPU node. SP2 distributed rolls cross
the node pair. DP reductions connect corresponding ranks in the two
model-parallel tiles. This is the unavoidable cost of requesting an eight-rank
model-parallel tile on four-GPU nodes.

## Expected Per-GPU Memory

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | `dp=1,tp=2,sp1=2,sp2=2` | 0.441 GB | 0.441 GB | 0.882 GB | ~7.986 GB | >=5.369 GB + unknown | ~0.300 GB | >=15.419 GB + unknown |
| 16 | `dp=2,tp=2,sp1=2,sp2=2` | 0.441 GB | 0.441 GB | 0.882 GB | ~3.993 GB | >=2.684 GB + unknown | ~0.300 GB | >=8.741 GB + unknown |
| 32 | `dp=4,tp=2,sp1=2,sp2=2` | 0.441 GB | 0.441 GB | 0.882 GB | ~1.997 GB | >=1.342 GB + unknown | ~0.300 GB | >=5.403 GB + unknown |
| 64 | `dp=8,tp=2,sp1=2,sp2=2` | 0.441 GB | 0.441 GB | 0.882 GB | ~0.998 GB | >=0.671 GB + unknown | ~0.300 GB | >=3.733 GB + unknown |

Local model state is about 110.220M parameter elements: TP shards block
linears, SP1/SP2 shard the position embedding, and the patch/head boundaries
remain replicated. Temporary memory includes a conservative allowance for
resident input, target, output, and loss-side tensors. The communication value
is a planning allowance, not a measured peak. Flash-SDPA workspace, CUDA
context, data-loader prefetching, NCCL internals, and allocator fragmentation
remain unknown.

## Scale Plan

Every supported row preserves:

```text
global_batch = microbatch_per_dp_rank * dp * accumulation = 16
```

| Scale | Model/input configuration | Mesh | Microbatch | Reason for changing model-parallel dimensions |
| --- | --- | --- | ---: | --- |
| 8 GPUs | Representative | `dp=1,tp=2,sp1=2,sp2=2` | 16 | Smallest tile with requested TP, SP1, and SP2 active |
| 16 GPUs, primary | Representative | `dp=2,tp=2,sp1=2,sp2=2` | 8 | Hold the model-parallel tile fixed and add DP |
| 32 GPUs | Representative | `dp=4,tp=2,sp1=2,sp2=2` | 4 | Throughput scale-out through DP |
| 64 GPUs | Representative | `dp=8,tp=2,sp1=2,sp2=2` | 2 | Throughput scale-out through DP |

The fixed global batch allows DP up to 16 with microbatch one and no
accumulation. Larger world sizes require a larger semantic global batch, a
different model-parallel tile, or idle devices; the launcher must not silently
change the batch.

## Pros, Cons, and Risks

Pros:

- SP1/SP2 reduce local image, token, and reconstruction activation sizes.
- Window attention stays local after a narrow, half-window distributed roll;
  it does not all-gather the full 4096-token sequence.
- TP halves block linear weights, heads, and MLP intermediates.
- The same eight-GPU model-parallel tile scales through DP.

Cons:

- The requested hybrid mesh adds communication even though pure DP fits.
- SP2 rolls cross nodes with four GPUs per node.
- Patch embedding and reconstruction-head parameters are replicated over TP.
- Alternating shifts, 2D masks, parameter shards, and extra wgrad reductions
  expand the correctness-test surface.

Risks:

- A local `torch.roll` would silently produce wrong windows at rank boundaries.
- A one-dimensional mask would leak attention across the other cyclic image
  boundary; the implementation must use the full 2D mask.
- Missing SP1-SP2 wgrad reductions silently change training.
- Local patch grids that are not divisible by the window cannot use the
  no-padding implementation and must be rejected.
- Runtime memory and throughput claims remain unverified until measured on the
  requested A100 allocation.

## Affected Files and Model Boundaries

- `networks/swin.py`: immutable single-device reference model.
- `networks/swin_mp.py`: TP/SP model, local shifted windows, and metadata.
- `distributed/helpers.py`: ring-neighbor distributed roll primitive.
- `distributed/mappings.py`: roll-forward/reverse-roll-backward autograd map.
- `distributed/spatial_sharding.py`: Swin window-alignment validation.
- `utils/comm.py`: complete planned group-token registry.
- `config/swin.yaml`: fixed architecture and existing training/data profiles.
- `train.py`, `train_mp.py`, `train_mp_graphs.py`: model dispatch without
  changing the default ViT path.
- `distributed/launch.sh`: validated SLURM `test`/`train` interface.
- Tests: distributed roll plus baseline/MP forward, input-gradient, and
  weight-gradient equivalence.

## Validation

1. Validate mesh products, exact process-group memberships, TP/head
   divisibility, and patch/window-aligned spatial splits.
2. Compare distributed roll with global `torch.roll`, including inverse
   input gradients, on SP1 and SP2.
3. Compare baseline and MP Swin with fixed seeds and dropout disabled for
   TP-only, SP1-only, SP2-only, and composed TP x SP1 x SP2 meshes.
4. Gather distributed outputs and input gradients over SP2, SP1, and DP and
   compare with the single-device reference.
5. Reconstruct each TP- and SP-sharded weight gradient and compare every
   trainable parameter with the reference.
6. Re-run existing ViT topology, spatial-sharding, and equivalence tests.
7. Measure CUDA/NVML peaks and samples per second; do not claim a runtime
   memory or throughput result without that measurement.

## Alternatives Considered

- **Pure DP:** simplest and likely fastest for this fixed model. Not selected
  because active TP, SP1, and SP2 were requested.
- **Global-attention ViT-style K/V gather:** reuses existing code but defeats
  Swin locality and increases attention FLOPs by 16x.
- **Local-only roll:** communication-free but incorrect at rank boundaries.
- **Patch merging / hierarchical Swin:** outside the requested quick,
  homogeneous architecture and would require stage-specific sharding.
- **FSDP:** rejected because most replicated parameters require non-DP
  SP1-SP2 wgrad reductions.
- **Sequence parallelism:** disabled because the explicit SP1/SP2 grid already
  addresses activation pressure and TP sequence transitions are unnecessary.
