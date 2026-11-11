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
