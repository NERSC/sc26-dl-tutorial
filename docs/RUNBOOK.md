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
