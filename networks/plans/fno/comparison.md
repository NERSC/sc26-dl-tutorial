# FNO: Plan Versus Implementation, and How General the Plan Method Is

This compares [the plan](parallelism-plan.md), obtained by applying the ViT plan's method to the
serial FNO ([analysis](analysis.md)), with the distributed implementation in
`networks/fno_mp.py` and `distributed/`. It answers two questions: does the implementation
do what the plan says, and did the ViT plan's method need extending?

## Verdict

The method carried over for the **parameter-metadata handoff, topology, validation and
alternatives** sections and for the overall workflow (analysis, then mesh, then per-op
sharding, then memory). It **did not carry over unchanged for the per-op communication table, the
memory/communication estimates, the scale plan and the cost model**, because the FNO has an
op (the pencil FFT) that changes layout and size inside a single logical op, it is
dominated by activations at full resolution, and it has complex parameters. Section 3 lists
eleven extensions. The implementation follows the plan; applying the plan also exposed three
defects in the implementation (Section 2), all fixed.

## 1. Plan versus implementation

| Plan item | Implementation | Status |
| --- | --- | --- |
| `sp1` shards H, `sp2` shards W, independently | `h_comm="sp1"`, `w_comm="sp2"` in `fno_mp.py` / `distributed/fft.py` | Matches |
| Each rank loads its (H, W) shard of input and target | PR #3's loaders with `spatial_shard` (`spatial_read_mode: local`) and `make_grid_spatial_shard`; the model takes the tile | Matches; `tests/test_sharded_io.py` (DALI itself not run: not installed here) |
| Lift and project after the scatter, replicated, `reduce: [sp1-sp2]` | `vit_mp.MLP(layout="channels_first")` on the shards; metadata `shared: [sp1-sp2]`, `reduce: [sp1-sp2]` | Matches |
| Spectral T1..T8 all-to-all sequence | `DistributedRealFFT2` / `DistributedInverseRealFFT2`, same order, uneven shards supported | Matches |
| Spectral weight `sharded: [null,null,sp1,sp2]`, `shared: [tp]`, `reduce: []` | `comm_metadata` in `FourierNeuralOperatorLayer` | Matches; verified by `tests/test_fno_model.py` (gradients at tp x sp1 x sp2 up to 2x2x2) and `tests/test_fno_ddp.py` |
| Skip layer replicated, `shared: [tp-sp1-sp2]`, `reduce: [sp1-sp2]` | `PointwiseLinear` | Matches |
| Loss and metrics on the shards, an all-reduce over SP1-SP2 of a scalar, no gather | PR #3's `mse_loss` and `rmse_spatial`, used unchanged by the FNO; the logging all-reduce stays over `dp` | Matches; equals the plain MSE of the full fields with the same gradients (`tests/test_sharded_io.py`, `tests/test_fno_model.py`). Changed from the first version, which gathered the output to the full image on every rank |
| General DDP hook (mixed reduction signatures) | `init_ddp_model_and_reduction_hooks`, unchanged | Matches; verified with complex parameters and `dp=2` on gloo (`tests/test_fno_ddp.py`) |
| Non-fused optimizer (fused Adam rejects complex weights) | Trainers hard-coded `fused=True` | **Differed, fixed**: `train.py` and `train_mp.py` now use `fused=False` if any parameter is complex |
| Divisibility: `H, W, kh, kw, C_in, C_out >= sp` | Only the channel count was checked, in the model; mode counts and spatial sizes failed at the first forward with a transposition error | **Differed, fixed**: checks at construction, in `distributed/fft.py` and `fno_mp.py` (`tests/test_fno.py::test_group_size_validation`) |
| Initialization statistically equivalent to the serial model, from global sizes | Spectral weights match. Pointwise weights use the ViT layers' constant `trunc_normal_(std=0.02)` and zero biases; the serial model uses PyTorch's conv default (fan-in scaled weights, random biases) | **Differs, fixed**: at width 16 the signal vanished at init (output std 0.0000, gradients 1e-13..1e-16) and the loss after 480 iterations was 0.596 versus 0.014 for the serial model. Now one rule, `init_pointwise_`, from global sizes in both models; `tests/test_fno_init.py`; `train_mp.py` trains (validation MSE 1.015 to 0.00025 in 960 iterations on the smoke problem) |
| Representative `embed_dim` | `config/FNO.yaml` `mp` carried 128 (2.27 TB of activations) | **Differed, fixed**: set to 16 |
| Activation checkpointing for the 4- and 8-GPU rows | Not implemented | Gap (the plan marks those rows infeasible) |
| `torch.compile` | The spectral layers are `@torch.compiler.disable`d in both models, so no FFT is ever in a compiled graph; the pointwise layers compile. The distributed model cannot be `fullgraph` anyway (collectives are excluded) | Matches, after a fix: with `enable_jit: true` on the NERSC container's torch the FFT fallback kernels failed an inductor stride assertion at 4096^2 (`expected stride 1==2049 at dim=2`, the c2c along h). Guarded by `test_compile_keeps_the_ffts_out_of_the_graph`. Not in the ViT plan's vocabulary |
| Memory and communication numbers | Analytical only | Not measured |

## 2. What applying the plan found

1. **Fused Adam crash.** `optim.Adam(..., fused=True)` raises on complex parameters, so
   `train_mp.py` and `train.py` would fail at optimizer creation on any FNO config. The ViT
   plan lists "Optimizer" as a contract field but has no compatibility check. Fixed.
2. **Late, cryptic validation errors.** `modes_kh < sp1` or `modes_kw < sp2` built fine and failed
   at the first forward pass with `cannot split dim 2 of size 3 into 4 chunks`. Fixed with
   construction-time checks.
3. **The initial `mp` config was unplannable** (2.27 TB of saved activations at batch 16). The
   width sweep in `analysis.md` made that visible only because the analysis computes cost
   per op.
4. **The distributed initialization is not equivalent to the serial one.** The model-parallel
   pointwise layers are reused from the ViT with their constant `std=0.02` and zero biases,
   while the serial FNO uses PyTorch's conv default. At `C=16` this vanishes the signal and
   slows training by more than an order of magnitude. The equivalence tests did not catch it
   because they copy the serial weights into the shards. Fixed: both models use `init_pointwise_` (std
   `1/sqrt(3 fan_in)`, biases `U(+-1/sqrt(fan_in))`) with the global fan-in, and
   `tests/test_fno_init.py` guards it, including the tensor-parallel cases where a local shape would
   give a wrong fan-in. The plan requires it (Initialization Handoff), with the measurements.
5. **`torch.compile` crashed at full size on another torch version.** The first cluster job with
   `enable_jit: true` died in the first forward pass with an inductor `assert_size_stride` error: the
   FFT fallback kernel returned a different memory layout than its meta kernel promises. The
   earlier compile checks (a `fullgraph=True` test on small images) passed on a newer torch and could
   not see it. Fixed by keeping the spectral layers out of the compiled graph, which does not depend
   on which FFT op or torch version is at fault. Not reproduced locally (torch 2.13 compiles at
   4096^2), so the fix is verified here only by the graph-contents test, and by the trainers running
   with `enable_jit: true`, not on the cluster's torch.
6. **Out of memory at 16 GPUs, from an fp32 residual stream and an optimistic estimate.** The first
   `mp` job (`dp=1, sp=4x4`, global batch 16) failed in the training forward with 38.5 GB in use.
   A single-rank probe at the per-GPU tile found the cause of part of it: the row-parallel bias
   (`x + bias`, a bf16 activation plus an fp32 parameter) promoted every block output of the distributed
   model to fp32, while the serial model stays in bf16, 23.9 GB kept against 21.8 GB after casting the
   bias. Fixed for the channels-first layout the FNO uses, guarded by two tests. The rest is the estimate
   itself: measured activations are 21.8 GB against 18.1 GB analytic, plus about 3 GB of transposition
   buffers and about 6 GB outside the PyTorch allocator. The `mp` config now uses global batch 8, and the
   plan carries the correction. The same promotion exists in the ViT's row-parallel layers, but there
   the residual stream is fp32 anyway (see below), and measured memory is identical.
7. **High-wavenumber spectral weights stay at their random init.** On band-limited data the gradient of the
   spectral weights falls by about 6 orders of magnitude from the lowest to the highest wavenumber band, the weights
   at high k barely move from their random values, and Adam's default `eps=1e-8` damps the update once the
   gradient is below it. Starting those weights smaller, `1/(1+(k/8)^2)`, improved the final validation loss in
   all 3 seeds (4.5e-5 to 3.2e-5), so it is on by default in `config/FNO.yaml`. A per-mode preconditioner
   (`W = D_k w`, `D_k = 1/S_k`) was tried and rejected: it made the validation loss 10x to 190x worse,
   because it let the weights of modes without signal random-walk. Any quantity that depends on the wavenumber has
   to use the **global** mode index in the distributed layer; this is tested, including that the test fails if the
   local index is used.
8. **Positive result:** the real DDP wrapper and reduction hooks handled complex parameters and
   mixed reduction signatures correctly (gradients equal the serial reference at `dp=2` with
   `sp1 x sp2 = 2x1, 2x2`, and with `tp=2`).

## 3. Where the plan method needed extending

Each item gives the extension and the evidence from the FNO.

1. **A 2D spatial mesh instead of a flattened sequence.** The ViT plan flattens `sp1 x sp2` into
   one group over the sequence and says that future 2D models may use the axes independently.
   This is that model. The `Logical`/`Sharded` columns are written in `b l e`; they need
   explicit grid dims `H W` and a statement of which mesh axis shards which dim.
2. **Layout-changing ops and the all-to-all collective.** The plan's collective vocabulary is
   scatter, gather, all-reduce and reduce-scatter. The spectral layer needs all-to-all
   transposes, and the sharding and shape change inside one logical op (spatial, then
   w-local, then spectral, with truncation in between). The table needs "op expansion"
   sub-rows (`T1..T8` here) with the shape and sharding after each step.
3. **Communication volume from the actual tensor at each step.** The ViT plan uses a flat
   `~0.3 GB` allowance. For the FNO, T1 and T8 move FP32 fields of 1.07 GB while the
   truncated transposes move about 0.03 GB, and send, receive and concatenation buffers triple
   the local field. The plan needs a per-transposition formula, and truncation must be
   modeled as a size change.
4. **Feasibility per scale row and activation checkpointing as a lever.** The ViT scale plan
   assumes every row fits. For the FNO the 4- and 8-GPU rows need 108 GB and 54 GB per GPU. The
   plan needs a feasibility column and, for activation-dominated models, checkpointing as a
   first-class option (here it would also allow wider models).
5. **A tensor-parallel profitability check.** TP is useful in the ViT because it shards large
   weights. In the FNO the pointwise weights total 24 KB while TP would all-reduce 6.4 GB of
   activations per step. A check comparing weight bytes saved with collective bytes added
   should come before TP is chosen.
6. **Mixed-precision islands and complex parameters in the contract.** The contract has one
   "activation/compute dtype". The FNO runs the spectral layers in FP32 under a BF16
   autocast and has complex64 parameters, which also change the "weights" and "optimizer"
   fields (and break fused Adam). The contract needs per-op dtypes and an
   optimizer-compatibility line.
7. **A cost model for FFT and complex ops.** The ViT tables mark non-matmul ops "Elementwise". The
   FFT and complex mode mixing need rows; the FFT is bandwidth-bound, so its FLOPs mislead.
   FFT speed also depends on layout: along strided dims it was 1.5-2x slower in our measurement
   on one GPU, which is the reason the FNO is channels-first.
8. **Restate the `reduce` rule.** The ViT plan says a reduction is needed "when the wgrad
   contraction covers only the rank's local sequence". The general rule, which also covers
   sharded spectral weights, is: a parameter needs a reduction over `G` iff it is replicated
   over `G` and the activations its wgrad contracts over are sharded over `G`. The
   `sharded`/`shared`/`reduce` format itself was sufficient, including weights sharded over
   spatial groups and complex dtypes. Mixed signatures also mean the fused single-group
   hook is unavailable and the general hook is used.
9. **Two kinds of divisibility constraints.** The ViT plan lists `% == 0` checks. The FNO
   supports uneven splits, but needs `>=` constraints on channels, retained modes and grid
   extents. The plan's divisibility row should state both kinds, and the implementation should
   check them at construction.

10. **Initialization as a plan item.** The ViT plan has no initialization section; the metadata
    handoff covers `shared` synchronization but not the distribution. The plan should require
    statistical (not bitwise) equivalence with the serial model: every scale computed from
    *global* sizes passed explicitly (a local `weight.shape` gives a wrong fan-in on
    row-parallel shards), biases matched as well as weights, the variance of complex
    parameters stated as `E|w|^2`, and constants inherited from another model justified at
    this width. It needs its own validation, since weight-copying equivalence tests cannot see
    it.

11. **The interface contract.** The ViT plan assumes a full image in and a full image out on every
    model-parallel rank, with a gather at the exit. For a model that works at full resolution
    that duplicates the input, target, output and loss difference on every rank of a replica,
    and the plan has no vocabulary for the alternative. The plan should state the interface per
    model (replicated or sharded) and what follows from it: the loader reads a shard, the loss
    reduces over the spatial group (the MSE needs the global sum, so one scalar is all-reduced, with
    an identity backward so that each rank gets the gradient of its own shard), metrics need the
    global pixel count, and plots need a gather. For the
    FNO the sharded interface saves about 2.8 GB per GPU at the primary mesh (27.0 to 24.2 GB),
    a 0.40 GB output gather per step, and every rank reads a quarter of the pixels.
Two further notes that do not change the method but belong in a shared template: the plan has
no compile-safety contract (which ops stay out of the graph, and a check on the graph contents rather
than on `fullgraph`, which passed here while the cluster's torch failed), and the
data contract (full image on every model-parallel rank, full gathered output) becomes a larger
cost for models with more channels, which motivates the sharded-loss alternative.

## 4. What is and is not verified

- **Verified (CPU, gloo, up to 8 ranks):** serial versus distributed outputs and every
  parameter gradient at `tp x sp1 x sp2` up to 2x2x2, the real DDP hooks at `dp=2`, uneven
  sizes, mode truncation, the self-conjugate zeroing against a CPU reference, the optimizer
  choice on one GPU.
- **Verified, sharded interface:** loader shards tile the image and match the full loader for
  the PyTorch loader and the DALI external source (called without DALI), in both read modes, and
  `mse_loss` and `rmse_spatial` on the shards equal the full ones with the same gradients at `sp1 x sp2` = 2x2, 4x1, 1x4 and with `tp=2`, and
  the model tests use shards in and out. `train_mp.py` ran end to end on one GPU with one rank
  for the FNO and the ViT (with stubs for TensorBoard, `pynvml` and `transformer_engine`, which are
  not installed here). The FNO trains through it: the validation MSE went from 1.015 to 0.00025 in
  960 iterations on the smoke problem (before the init fix it stayed at exactly 1.0).
- **Verified, initialization:** `tests/test_fno_init.py` at `tp x sp1 x sp2` up to 2x2x2 (statistics,
  synchronization of replicated parameters, signal and gradient scale against the serial model over
  15 draws), and it fails with the old `std=0.02`. The five-variant comparison that found the problem
  is one seed and 480 iterations, indicative only. Whether this scale is *critical* for signal
  propagation (the literature on FNO initialization) is not studied here.
- **Not verified:** a multi-rank `train_mp.py` run (one GPU here, and `comm.init` uses NCCL), the DALI pipeline itself,
  NCCL and multi-GPU runs (the `dist.all_to_all` path), peak memory,
  communication time, throughput, an end-to-end training step with the data loader, and every
  number in `analysis.md` and the plan, which are closed-form estimates.
