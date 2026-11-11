# Tutorial runbook

`sbatch` commands that reproduce the README figures. Each command requests the job itself.

Training arguments stay on the command line and are forwarded to `train.py` or `train_mp.py`. `ENABLE_PROFILING=1` profiles every rank. Reports land in `$SCRATCH/sc26-dl-tutorial/logs/` as `<name>.<rank>.nsys-rep`. Training logs are under that same directory, mounted in the container as `/logs`.

Override `DATADIR` or `LOGDIR` in the environment if the defaults in the scripts are not the ones you want.

## ViT

### 1. Baseline — `short`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short --run_num short
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short/1GPU/short/out.log`

### 2. Workers, BF16, fused Adam, torch.compile — `short_opt`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short_opt --run_num short_opt
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt/1GPU/short_opt/out.log`

### 3. DALI — `short_opt_dali`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/ViT.yaml --config short_opt_dali --run_num dali
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali/1GPU/dali/out.log`

### 4. Data parallel — `short_opt_dali_dp`, 4 GPUs

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_dp \
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/ViT.yaml --config short_opt_dali_dp --run_num dp
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali_dp/4GPU/dp/out.log`

Profiles: `vit_dp.0.nsys-rep` through `vit_dp.3.nsys-rep`.

### 5. Large model, local batch 1 — `short_opt_big`, 4 GPUs

Same 4 GPUs as the first data-parallel run. `global_batch_size: 4` makes the local batch 1. The ViT is larger: `patch_size: 16`, `embed_dim: 1024`. This one fits. No profile.

```bash
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/ViT.yaml --config short_opt_big --run_num big
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big/4GPU/big/out.log`

### 6. Large model, 16 GPUs — `short_opt_big_dp16`

Same model and the same local batch of 1. The global batch is 16, and `lr` is `1e-3`, from `5e-4 * sqrt(16 / 4)`. No profile.

```bash
sbatch -N 4 ./submit_dp.sh --yaml_config ./config/ViT.yaml --config short_opt_big_dp16 --run_num big16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/big16/out.log`

### 7. Out of memory, model parallel with no split — `short_mp`, 4 GPUs

`embed_dim: 1536` and `depth: 16`, local batch still 1. `--tp=1 --sp1=1 --sp2=1` leaves the whole replica on every GPU, so this is the out-of-memory run. Profiling stays off.

```bash
sbatch -N 1 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp --run_num oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/1MP/oom/out.log`

### 8. Model parallel — `short_mp`, 16 GPUs

The same config as the failed run, with the global batch still 4. `sp1=4` shards the token sequence inside each node. Four nodes leave `dp=4`, so the local batch stays 1.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp --run_num mp \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/4MP/mp/out.log`

Profiles: `vit_mp.0.nsys-rep` through `vit_mp.15.nsys-rep`.

### 9. Transformer Engine, sequence sharded — `short_mp_te`, 16 GPUs

Same model and mesh as the spatial run. `short_mp_te` sets `model_backend: transformer-engine`. `NVTE_BATCH_MHA_P2P_COMM=1` batches the context-parallel K/V send and recv. `sp1=4` shards the sequence inside the node, and `dp=4` keeps the local batch at 1.

```bash
NVTE_BATCH_MHA_P2P_COMM=1 \
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp_te \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_te --run_num mp_te \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp_te/4MP/mp_te/out.log`

Profiles: `vit_mp_te.0.nsys-rep` through `vit_mp_te.15.nsys-rep`.

### 10. Transformer Engine, tensor parallel — `short_mp_te`, 16 GPUs

Same config, with the sequence left whole. `tp=4` shards the heads and the MLP across the four GPUs in the node. `1536` and the 16 heads divide by 4. Four nodes leave `dp=4`, so the local batch stays 1. `CUDA_DEVICE_MAX_CONNECTIONS=1` keeps the backward all-reduce ahead of the weight-gradient GEMM so the two overlap.

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_mp_te_tp \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_te --run_num mp_te_tp \
    --tp=4 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp_te/4MP/mp_te_tp/out.log`

Profiles: `vit_mp_te_tp.0.nsys-rep` through `vit_mp_te_tp.15.nsys-rep`.

### 11. Activation checkpointing — `short_mp_ckpt`, 4 GPUs

Same model as the out-of-memory run, with `activation_checkpoint: true`. `--tp=1 --sp1=1 --sp2=1` leaves the replica whole on one node.

```bash
sbatch -N 1 ./submit_mp.sh --yaml_config ./config/ViT.yaml --config short_mp_ckpt --run_num ckpt \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp_ckpt/1MP/ckpt/out.log`


---

## Swin

Same sizes as the ViT at each step, except Swin skips the local-batch-4 data-parallel run. The first data-parallel run is the larger model at local batch 1. `window_size: [16, 16]` stays, and 16 divides the patch grid of every step below. `run_num` is prefixed with `swin` because the log path is `<config>/<N>GPU/<run_num>` and does not include the YAML file name.

### 1. Baseline — `short`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short --run_num swin
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short/1GPU/swin/out.log`

### 2. Workers, BF16, fused Adam, torch.compile — `short_opt`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short_opt --run_num swin_opt
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt/1GPU/swin_opt/out.log`

### 3. DALI — `short_opt_dali`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/swin.yaml --config short_opt_dali --run_num swin_dali
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali/1GPU/swin_dali/out.log`

### 4. Large model, local batch 1 — `short_opt_big`, 4 GPUs

`patch_size: 16` and `embed_dim: 1024`, with `depth` still 12. The patch grid is 256×256, and `window_size` 16 divides it. Global batch 4 on 4 GPUs, so the local batch is 1.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_short_opt_dali_big \
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big --run_num swin_big_profile
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big/4GPU/swin_big_profile/out.log`

Profiles: `swin_short_opt_dali_big.0.nsys-rep` through `swin_short_opt_dali_big.3.nsys-rep`.

### 5. Large model, 16 GPUs — `short_opt_big_dp16`

Same model and the same local batch of 1. The global batch is 16, and `lr` is `1e-3`, from `5e-4 * sqrt(16 / 4)`. No profile.

```bash
sbatch -N 4 ./submit_dp.sh --yaml_config ./config/swin.yaml --config short_opt_big_dp16 --run_num swin_big16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/swin_big16/out.log`

### 6. Out of memory, model parallel with no split — `short_mp`, 4 GPUs

`patch_size: 16`, `embed_dim: 1536`, and `depth: 16`, local batch still 1. The patch grid is 256×256, which 16 divides. `--tp=1 --sp1=1 --sp2=1` leaves the whole replica on every GPU. This is the same size-up that ran out of memory for the ViT. Profiling stays off.

```bash
sbatch -N 1 ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/1MP/swin_oom/out.log`

### 7. Model parallel — `short_mp`, 16 GPUs

The same config, with the global batch still 4. `sp1=4` shards one spatial axis. The local grid is 64×256, and window 16 divides both sides. Four nodes leave `dp=4`, so the local batch stays 1.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_mp \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_mp \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/4MP/swin_mp/out.log`

Profiles: `swin_mp.0.nsys-rep` through `swin_mp.15.nsys-rep`.

---

## FNO

Same steps as the Swin. The widths are parameter-matched to the ViT, and `depth` stays 4 with modes 32×32. The small model is `embed_dim: 76` (47.4M) on a 1024×1024 latent grid (`downsample: 4`), with global batch 1. The resample FFT keeps the full-grid lift in FP32, so batch 4 does not fit under BF16 either. The large model is `embed_dim: 96` (75.9M) on a 1024×1024 latent grid (`downsample: 4`). `short_mp` is `embed_dim: 260` (555.1M) on a 2048 grid (`downsample: 2`), sharded over `sp1 × sp2`. `enable_fused` stays off, because fused Adam cannot update the complex spectral weights. `run_num` is prefixed with `fno` because the log path is `<config>/<N>GPU/<run_num>` and does not include the YAML file name.

### 1. Baseline — `short`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short --run_num fno
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short/1GPU/fno/out.log`

### 2. Workers, BF16, torch.compile — `short_opt`, 1 GPU

Two workers, BF16, and `torch.compile`. The loader is still PyTorch. Fused Adam stays off.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short_opt \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short_opt --run_num fno_opt
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt/1GPU/fno_opt/out.log`

### 3. DALI — `short_opt_dali`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_short_opt_dali \
sbatch -n 1 ./submit.sh --yaml_config ./config/FNO.yaml --config short_opt_dali --run_num fno_dali
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali/1GPU/fno_dali/out.log`

### 4. Large model, local batch 1 — `short_opt_big`, 4 GPUs

`downsample: 4` and `embed_dim: 96`. The latent grid is 1024×1024. `embed_dim: 163` does not fit at local batch 1. Global batch 4 on 4 GPUs, so the local batch is 1.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_big \
sbatch -N 1 ./submit_dp.sh --yaml_config ./config/FNO.yaml --config short_opt_big --run_num fno_big
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big/4GPU/fno_big/out.log`

Profiles: `fno_big.0.nsys-rep` through `fno_big.3.nsys-rep`.

### 5. Large model, 16 GPUs — `short_opt_big_dp16`

Same model and the same local batch of 1. The global batch is 16, and `lr` is `1e-3`, from `5e-4 * sqrt(16 / 4)`. No profile.

```bash
sbatch -N 4 ./submit_dp.sh --yaml_config ./config/FNO.yaml --config short_opt_big_dp16 --run_num fno_big16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/fno_big16/out.log`

### 6. Out of memory, model parallel with no split — `short_mp`, 4 GPUs

`embed_dim: 260` and `downsample: 2`, local batch still 1. The latent grid is 2048×2048. `--tp=1 --sp1=1 --sp2=1` leaves the whole field on every GPU. Profiling stays off.

```bash
sbatch -N 1 ./submit_mp.sh --yaml_config ./config/FNO.yaml --config short_mp --run_num fno_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/1MP/fno_oom/out.log`

### 7. Model parallel — `short_mp`, 16 GPUs

The same config, with the global batch still 4. `sp1=2` and `sp2=2` shard height and width inside each node, so the pencil-FFT all-to-alls stay on NVLink. Four nodes leave `dp=4`, and the local batch stays 1. Each rank's latent tile is 1024×1024. `--tp=1`.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_mp \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/FNO.yaml --config short_mp --run_num fno_mp \
    --tp=1 --sp1=2 --sp2=2
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/4MP/fno_mp/out.log`

Profiles: `fno_mp.0.nsys-rep` through `fno_mp.15.nsys-rep`.

## ConvNeXt

The single-GPU ladder uses patch 32, width 512, depth 12, kernel 7, and batch 4.
Data parallelism starts at local batch 4, then uses a larger model at local
batch 1. Spatial parallelism uses patch 16 and width 768, with `tp=1`.

### 1. Baseline — `short`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short \
sbatch -q debug -N 1 -n 1 --ntasks-per-node=1 --time=00:30:00 -J convnext_short \
    ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short --run_num convnext
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short/1GPU/convnext/out.log`

### 2. Workers, BF16, fused Adam, torch.compile — `short_opt`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short_opt \
sbatch -q debug -N 1 -n 1 --ntasks-per-node=1 --time=00:30:00 -J convnext_short_opt \
    ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt --run_num convnext_opt
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt/1GPU/convnext_opt/out.log`

### 3. DALI — `short_opt_dali`, 1 GPU

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_short_opt_dali \
sbatch -q debug -N 1 -n 1 --ntasks-per-node=1 --time=00:30:00 -J convnext_short_opt_dali \
    ./submit.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_dali --run_num convnext_dali
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali/1GPU/convnext_dali/out.log`

### 4. Data parallel — `short_opt_dali_dp4`, 4 GPUs

Global batch 16, local batch 4. The learning rate is `1e-3`, from
`5e-4 * sqrt(16 / 4)`. `num_iters: 80` schedules five epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_dp4 \
sbatch -q debug -N 1 -n 4 --ntasks-per-node=4 --time=00:30:00 -J convnext_dp4 \
    ./submit_dp.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_dali_dp4 --run_num convnext_dp4
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali_dp4/4GPU/convnext_dp4/out.log`

Profiles: `convnext_dp4.0.nsys-rep` through `convnext_dp4.3.nsys-rep`.

### 5. Data parallel — `short_opt_dali_dp16`, 16 GPUs

Global batch 64, local batch 4. The learning rate is `2e-3`, from
`5e-4 * sqrt(64 / 4)`. `num_iters: 20` schedules five epochs. This run uses
128 validation samples so each rank has two batches; the other runs use 32.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_dp16 \
sbatch -q debug -N 4 -n 16 --ntasks-per-node=4 --time=00:30:00 -J convnext_dp16 \
    ./submit_dp.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_dali_dp16 --run_num convnext_dp16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_dali_dp16/16GPU/convnext_dp16/out.log`

Profiles: `convnext_dp16.0.nsys-rep` through `convnext_dp16.15.nsys-rep`.

### 6. Large model, local batch 1 — `short_opt_big`, 4 GPUs

Patch 16, width 1024, depth 12, kernel 7. Global batch 4 makes the local batch
1. BF16, fused Adam, compilation, and DALI with four workers remain enabled.
The run uses 256 training and 32 validation samples, with 320 iterations for
five epochs. Check memory use and GPU activity in this run before scaling out.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_big \
sbatch -q debug -N 1 -n 4 --ntasks-per-node=4 --time=00:30:00 -J convnext_big \
    ./submit_dp.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_big --run_num convnext_big
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big/4GPU/convnext_big/out.log`

Profiles: `convnext_big.0.nsys-rep` through `convnext_big.3.nsys-rep`.

### 7. Large model, 16 GPUs — `short_opt_big_dp16`

Same model and local batch 1. Global batch 16, learning rate `1e-3`, from
`5e-4 * sqrt(16 / 4)`. The sample counts stay the same; 80 iterations schedule
five epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_big16 \
sbatch -q debug -N 4 -n 16 --ntasks-per-node=4 --time=00:30:00 -J convnext_big16 \
    ./submit_dp.sh --yaml_config ./config/ConvNeXt.yaml --config short_opt_big_dp16 --run_num convnext_big16
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_opt_big_dp16/16GPU/convnext_big16/out.log`

Profiles: `convnext_big16.0.nsys-rep` through `convnext_big16.15.nsys-rep`.

### 8. Out of memory, no spatial split — `short_mp`, 1 GPU

Patch 16, width 768, depth 12, batch 4. This uses `train_mp.py` with no split,
BF16, fused Adam, DALI with two workers, and eager execution. Profiling is off.
The training forward runs out of memory in the channel MLP's first linear layer.

```bash
ENABLE_PROFILING=0 \
sbatch -q debug -N 1 -n 1 --ntasks-per-node=1 --gpus-per-node=1 --gpu-bind=none \
    --time=00:30:00 -J convnext_unsplit ./submit_mp.sh \
    --yaml_config ./config/ConvNeXt.yaml --config short_mp --run_num convnext_unsplit \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/1MP/convnext_unsplit/out.log`

### 9. Spatial parallel — `short_mp`, 4 GPUs

The same model and batch, split across four GPUs on one node. `sp1=4` divides
the 256×256 feature grid into 64×256 tiles. Weights remain replicated.
This run fits and completes training and validation.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_sp4 \
sbatch -q debug -N 1 -n 4 --ntasks-per-node=4 --time=00:30:00 -J convnext_sp4 \
    ./submit_mp.sh --yaml_config ./config/ConvNeXt.yaml --config short_mp --run_num convnext_sp4 \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/4MP/convnext_sp4/out.log`

Profiles: `convnext_sp4.0.nsys-rep` through `convnext_sp4.3.nsys-rep`.

### 10. Spatial and data parallel — `short_mp_dp4`, 16 GPUs

Four nodes hold four spatially sharded replicas (`dp=4`). Local batch stays 4,
global batch becomes 16, and `lr=1e-3`. `num_iters: 80` schedules five epochs.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=convnext_sp4_dp4 \
sbatch -q debug -N 4 -n 16 --ntasks-per-node=4 --time=00:30:00 -J convnext_sp4_dp4 \
    ./submit_mp.sh --yaml_config ./config/ConvNeXt.yaml --config short_mp_dp4 --run_num convnext_sp4_dp4 \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp_dp4/4MP/convnext_sp4_dp4/out.log`

Profiles: `convnext_sp4_dp4.0.nsys-rep` through `convnext_sp4_dp4.15.nsys-rep`.

## After a run

The throughput lines are in the training log and in the job output. Copy the `.nsys-rep` files into `profiles/` so the reports stay with the tutorial. Save the Nsight screenshot as:

| Run | Image |
|---|---|
| `short` | `tutorial_images/vit_short_timeline.png`, `tutorial_images/vit_short_step.png` |
| `short_opt` | `tutorial_images/vit_short_opt.png`, `tutorial_images/vit_short_opt_zoomed.png` |
| `short_opt_dali` | `tutorial_images/vit_short_opt_dali.png` |
| `short_opt_dali_dp` | `tutorial_images/vit_dp.png` |
| `short_opt_big` | `tutorial_images/vit_big.png` |
| `short_mp` | `tutorial_images/vit_mp.png` |
| `short_mp_te` | `tutorial_images/vit_te_sp.png` |
| `short_mp_te` `tp=4` | `tutorial_images/vit_te_tp.png` |
| `swin` `short` | `tutorial_images/swin_short_step.png` |
| `swin` `short_opt` | `tutorial_images/swin_short_opt.png` |
| `swin` `short_opt_dali` | `tutorial_images/swin_short_opt_dali.png` |
| `swin` `short_opt_big` | `tutorial_images/swin_big.png` |
| `swin` `short_mp` | `tutorial_images/swin_mp.png` |
| `fno` `short` | `tutorial_images/fno_short_step.png` |
| `fno` `short_opt` | `tutorial_images/fno_short_opt.png` |
| `fno` `short_opt_dali` | `tutorial_images/fno_short_opt_dali.png` |
| `fno` `short_opt_big` | `tutorial_images/fno_big.png` |
| `fno` `short_mp` | `tutorial_images/fno_mp.png` |
| `convnext` `short` | `tutorial_images/convnext_short_step.png` |
| `convnext` `short_opt` | `tutorial_images/convnext_short_opt.png` |
| `convnext` `short_opt_dali` | `tutorial_images/convnext_short_opt_dali.png` |
| `convnext` `short_opt_dali_dp4` | `tutorial_images/convnext_dp4.png` |
| `convnext` `short_opt_dali_dp16` | `tutorial_images/convnext_dp16.png` |
| `convnext` `short_opt_big` | `tutorial_images/convnext_big.png` |
| `convnext` `short_opt_big_dp16` | `tutorial_images/convnext_big16.png` |
| `convnext` `short_mp` | `tutorial_images/convnext_sp4.png` |
| `convnext` `short_mp_dp4` | `tutorial_images/convnext_sp4_dp4.png` |
