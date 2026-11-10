# Swin: model parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md)

The mesh is the same one documented on the [ViT model-parallel page](../vit/model-parallel.md): `dp × tp × sp1 × sp2`, with column-parallel and row-parallel linears over `tp`. Swin keeps `sp1` and `sp2` as separate spatial axes. A block of windows lives on each spatial rank, and the shifted-window blocks exchange the cells that cross a rank boundary through `roll_forward_reverseroll_backward` in [`networks/swin_mp.py`](../../networks/swin_mp.py). `window_size` has to divide the patch grid on each spatial rank. The interactive commands are steps 7 and 8 under Swin in [RUNBOOK_run.md](../RUNBOOK_run.md).

## No split

`short_mp` is the same size-up that ran out of memory for the ViT: `patch_size: 16`, `embed_dim: 1536`, `depth: 16`, local batch 1. The patch grid is 256×256, which 16 divides. `--tp=1 --sp1=1 --sp2=1` leaves the whole replica on every GPU. Window attention stores less activation memory than full attention, so this run is the check of whether that is enough. Profiling stays off.

```bash
sbatch -N 1 ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/1MP/swin_oom/out.log`.

> **TODO:** local parameter count, and either the forward memory or the out-of-memory error.

## Spatial split

`sp1=4` shards one spatial axis. The local grid is 64×256, and window 16 divides both sides. The shift on odd blocks is half a window, 8, which is inside that local grid. Four nodes leave `dp=4`, so the global batch of 4 keeps the local batch at 1. A group of size 1 on `sp2` rolls locally.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=swin_mp \
sbatch -N 4 ./submit_mp.sh --yaml_config ./config/swin.yaml --config short_mp --run_num swin_mp \
    --tp=1 --sp1=4 --sp2=1
```

Log: `$SCRATCH/sc26-dl-tutorial/logs/short_mp/4MP/swin_mp/out.log`. Profiles: `swin_mp.0.nsys-rep` through `swin_mp.15.nsys-rep`.

> **TODO:** first three epochs, peak memory, and samples/sec.

> **TODO:** timeline of the shifted-window exchange in the forward and the backward.
