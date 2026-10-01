# ViT TP + CP Parallelism Plan

This plan uses the fixed model contract in [analysis.md](analysis.md). The
primary target is 16 NVIDIA 40 GB GPUs, with four NVLink-connected GPUs per
node, global batch size 16, and a throughput objective.

## Constraints and Candidate Mesh

| Field | Value |
| --- | --- |
| Representative model configuration | 12 layers, `e=1024`, `h=16`, `f=4096`, `l=4096` |
| Memory target per device | 40 GB |
| Global batch, microbatch, accumulation | Primary: `16 = 4 x dp(4) x 1` |
| Representative input | `(16, 4, 4096, 4096)` globally |
| Available devices | 16 primary; projections for 4 and 8 |
| Hardware topology | Four GPUs per node with NVLink |
| Bottleneck | Saved activations; attention and MLP compute |
| Candidate mesh | Primary: `dp=4, tp=2, cp=2` |
| Data-parallel wrapper | DDP with CP wgrad-reduction hook |
| Sequence parallelism | `false`; CP already shards sequence |

Pure DP likely provides higher throughput and already fits 40 GB after local
batch scaling. TP and CP are active because both were explicitly requested.
The minimum active hybrid tile is therefore `tp=2, cp=2`; remaining GPUs are
assigned to DP.

## Operation Sharding and Communication

For the primary mesh, local batch `b_dp=4`, context length `l_cp=2048`, local
heads `h_tp=8`, and local MLP width `f_tp=2048`.

| Op group | Mesh | Logical | Sharded | Forward collectives | dgrad collectives | wgrad collectives |
| --- | --- | --- | --- | --- | --- | --- |
| Patch + position | `dp x tp x cp` | `b l e` | `b/dp l e`, then `b/dp l/cp e` | Scatter over CP | Gather over CP | DDP only; MP copies are duplicate |
| LayerNorm | `dp x tp x cp` | `b l e` | `b/dp l/cp e` | None | None | All-reduce CP for affine parameters |
| Q/K/V, column TP | `dp x tp x cp` | `b l e, e e -> b l e` | `b/dp l/cp e, e e/tp -> b/dp l/cp e/tp` | None | All-reduce TP | All-reduce CP; DDP over DP |
| Attention core | `dp x tp x cp` | `b h l d` | Q: `b/dp h/tp l/cp d`; K/V gathered to `l` | All-gather K/V over CP | Reduce-scatter dK/dV over CP | None |
| Attention projection, row TP | `dp x tp x cp` | `b l e, e e -> b l e` | `b/dp l/cp e/tp, e/tp e -> b/dp l/cp e` | All-reduce TP | None | All-reduce CP; DDP over DP |
| MLP FC1, column TP | `dp x tp x cp` | `b l e, e f -> b l f` | `b/dp l/cp e, e f/tp -> b/dp l/cp f/tp` | None | All-reduce TP | All-reduce CP; DDP over DP |
| MLP FC2, row TP | `dp x tp x cp` | `b l f, f e -> b l e` | `b/dp l/cp f/tp, f/tp e -> b/dp l/cp e` | All-reduce TP | None | All-reduce CP; DDP over DP |
| Final LayerNorm | `dp x tp x cp` | `b l e` | `b/dp l/cp e` | None | None | All-reduce CP for affine parameters |
| CP exit + head | `dp x tp x cp` | `b l e -> b 4 H W` | Gather to `b/dp l e`; replicated head | All-gather CP | Split over CP | DDP only; MP copies are duplicate |

The CP attention implementation gathers K and V but keeps Q context-local.
This preserves global attention semantics while reducing query/output
activation storage by CP degree. Its backward mapping reduce-scatters dK and
dV.

## Parameter Metadata Handoff

Logical operation shapes use mathematical weight order. Parameter and local
shapes use actual PyTorch storage order.

| Parameter pattern | Logical op shape | Logical local shape | Parameter shape | Local parameter shape | sharded | shared | reduce | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_embed.proj.weight` | Conv kernel | Same | `(1024,4,64,64)` | Same | `[null,null,null,null]` | `["tp-cp"]` | `[]` | Patch embedding executes before CP scatter and is replicated |
| `patch_embed.proj.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `[]` | Replicated model-parallel copy |
| `pos_embed` | `l e` | Same | `(1,4096,1024)` | Same | `[null,null,null]` | `["tp-cp"]` | `[]` | Added before CP scatter |
| `blocks.*.norm*.weight` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `["cp"]` | CP ranks own different sequence positions |
| `blocks.*.norm*.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `["cp"]` | CP ranks own different sequence positions |
| `blocks.*.attn.{q,k,v}.weight` | `e e` | `e e/tp` | `(1024,1024)` | `(512,1024)` | `["tp",null]` | `["cp"]` | `["cp"]` | Output-feature TP shard; CP-local contracting sequence |
| `blocks.*.attn.{q,k,v}.bias` | `e` | `e/tp` | `(1024)` | `(512)` | `["tp"]` | `["cp"]` | `["cp"]` | Output-feature TP shard |
| `blocks.*.attn.proj.weight` | `e e` | `e/tp e` | `(1024,1024)` | `(1024,512)` | `[null,"tp"]` | `["cp"]` | `["cp"]` | Input-feature TP shard; CP-local sequence |
| `blocks.*.attn.proj.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `["cp"]` | Row-parallel bias is replicated over TP |
| `blocks.*.mlp.fc1.weight` | `e f` | `e f/tp` | `(4096,1024)` | `(2048,1024)` | `["tp",null]` | `["cp"]` | `["cp"]` | Output-feature TP shard |
| `blocks.*.mlp.fc1.bias` | `f` | `f/tp` | `(4096)` | `(2048)` | `["tp"]` | `["cp"]` | `["cp"]` | Output-feature TP shard |
| `blocks.*.mlp.fc2.weight` | `f e` | `f/tp e` | `(1024,4096)` | `(1024,2048)` | `[null,"tp"]` | `["cp"]` | `["cp"]` | Input-feature TP shard |
| `blocks.*.mlp.fc2.bias` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `["cp"]` | Row-parallel bias is replicated over TP |
| `norm.{weight,bias}` | `e` | Same | `(1024)` | Same | `[null]` | `["tp-cp"]` | `["cp"]` | CP-local sequence |
| `head.weight` | `e o` | Same | `(16384,1024)` | Same | `[null,null]` | `["tp-cp"]` | `[]` | Head executes after CP gather and is replicated |

Ordinary DP wgrad synchronization is handled by DDP and is not included in
`reduce`. A CP reduction is required when the wgrad contraction covers only
the rank's local context. Replicated TP copies that calculate identical,
rather than partial, gradients are not TP-reduced.

## Topology Handoff

| Field | Planned value |
| --- | --- |
| World size | 16 |
| Launch method | `srun -N 4 --ntasks-per-node=4 --gpus-per-node=4 ... train_mp.py` |
| Mesh dimensions | `dp=4, tp=2, cp=2` |
| Sequence parallelism | `false` |
| Rank order | `tp-cp-dp` |
| Group tokens to build | `dp`, `tp`, `cp`, `tp-cp`, `dp-tp`, `dp-cp`, `dp-tp-cp` |
| Hardware topology | One contiguous four-rank TP x CP tile per NVLink node |
| Divisibility | `1024 % 2 == 0`; `4096 % 2 == 0`; `16 % 2 == 0`; no padding |

The world-size invariant is:

```text
16 = dp(4) * tp(2) * cp(2)
```

With `tp-cp-dp`, TP ranks vary fastest. Node 0 contains ranks 0-3:

```text
TP groups: (0,1), (2,3)
CP groups: (0,2), (1,3)
example DP group: (0,4,8,12)
```

Thus all TP and CP communication remains within NVLink nodes; DP
gradient synchronization spans nodes.

The complete token set includes atomic dimensions, `tp-cp` used for replicated
model-parallel parameters, and the required DP-composite candidates for
non-DP `shared`/`reduce` tokens. The current `utils/comm.py` does not build
`dp-tp` or `dp-tp-cp`; those groups must be added before treating this as the
final topology implementation.

## Expected Per-GPU Memory

| GPUs | Mesh | Weights | Gradients | Optimizer | Saved activations | Temporary | Communication | Total per GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `dp=1,tp=2,cp=2` | 0.453 GB | 0.453 GB | 0.907 GB | 15.972 GB | >=10.737 GB + unknown | ~0.300 GB | >=28.823 GB + unknown |
| 8 | `dp=2,tp=2,cp=2` | 0.453 GB | 0.453 GB | 0.907 GB | 7.986 GB | >=5.369 GB + unknown | ~0.300 GB | >=15.469 GB + unknown |
| 16 | `dp=4,tp=2,cp=2` | 0.453 GB | 0.453 GB | 0.907 GB | 3.993 GB | >=2.684 GB + unknown | ~0.300 GB | >=8.791 GB + unknown |

Temporary memory includes resident FP32 input and target tensors plus a BF16
model output at the local DP batch. The 0.3 GB communication value is a
planning allowance, not a measured peak. Flash-SDPA workspace, CUDA context,
DALI prefetching, NCCL internals, and allocator fragmentation remain unknown.

## Scale Plan

For every row:

```text
global_batch = microbatch_per_dp_rank * dp * accumulation = 16
```

| Scale | Model/input configuration | Mesh | Microbatch | Reason for changing model-parallel dimensions |
| --- | --- | --- | ---: | --- |
| 4 GPUs | Representative | `dp=1,tp=2,cp=2` | 16 | Smallest mesh with TP and CP active |
| 8 GPUs | Representative | `dp=2,tp=2,cp=2` | 8 | Hold TP/CP fixed and add DP |
| 16 GPUs, primary | Representative | `dp=4,tp=2,cp=2` | 4 | Hold TP/CP fixed and add DP across four nodes |

## Bottleneck and Rationale

Saved activations dominate the unsharded first-order state. CP halves
transformer sequence activations and the local attention-query extent. TP
halves attention heads, transformer linear weights, and MLP intermediates.
Keeping `tp=2, cp=2` within each four-GPU node places the frequent
model-parallel collectives on NVLink, while DP is the throughput scale-out
axis.

## Pros, Cons, and Risks

Pros:

- TP reduces transformer linear weights and wide intermediates.
- CP reduces block sequence activations and attention-query compute.
- TP and CP communication stays within each NVLink node.
- The same four-GPU MP tile scales from 4 to 16 GPUs through DP.

Cons:

- Patch embedding and reconstruction head work is repeated over TP/CP ranks.
- Every block adds TP collectives and CP K/V exchange.
- Hybrid rank and gradient bookkeeping is more complex than pure DP.
- Global batch 16 limits useful DP scaling beyond 16 DP ranks.

Risks:

- Non-memory-efficient SDPA may materialize very large attention matrices.
- Missing CP wgrad reductions silently produce incorrect training.
- Runtime memory can exceed static estimates due to DALI, kernels, NCCL, and
  allocator behavior.
- The current topology registry lacks two derived DP-composite group tokens.

## Affected Files and Model Boundaries

- `utils/comm.py`: validate the mesh product and build the complete group set.
- `networks/vit_mp.py`: TP linear and CP attention contracts and metadata.
- `distributed/mappings.py`: autograd collectives and DDP reduction hooks.
- `networks/vit.py`: CP entry/exit and distributed block selection.
- `train_mp.py`: mesh CLI, batch invariant, DDP wrapper, and launch logging.
- Tests: topology and numerical-equivalence coverage.

## Validation

1. Test pure rank generation and exact memberships for every group token at
   4, 8, and 16 GPUs.
2. Compare baseline, TP-only, CP-only, and composed TP x CP forward outputs
   with fixed seeds and dropout disabled.
3. Compare reconstructed input gradients against the baseline.
4. Compare every local weight-gradient shard after CP and DP reductions.
5. Verify replicated patch, position, normalization, bias, and head parameters
   remain synchronized.
6. Measure peak allocated/reserved CUDA memory and NVML high-water marks.
7. Benchmark samples/second and collective traces against pure DP; do not
   claim a throughput improvement without measurement.

## Alternatives Considered

- **Pure DP:** Simpler, avoids model-parallel collectives, already fits 40 GB
  after local-batch scaling, and is likely faster. Rejected only because active
  TP and CP were requested.
- **Larger TP or CP:** Adds communication without solving a remaining capacity
  requirement for this fixed model and input.
- **FSDP:** The first-order path is invalid because transformer parameters have
  non-DP CP wgrad reductions; it would require a tested, version-specific
  communication hook.
- **Sequence parallelism:** Disabled because CP already addresses sequence
  activation pressure and an additional TP layout transition is not justified.
