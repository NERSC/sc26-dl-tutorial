# FNO: model parallelism

[Tutorial](../../README.md) · [Single GPU](single-gpu.md) · [Data parallel](data-parallel.md)

The mesh is the same one documented on the [ViT model-parallel page](../vit/model-parallel.md): `dp × tp × sp1 × sp2`. The FNO keeps `sp1` and `sp2` as separate spatial axes. Height is sharded over `sp1` and width over `sp2`. The spectral layer is a pencil FFT: `distributed_transpose` swaps which axis is sharded, the retained modes are mixed, and the inverse transpose comes back in the backward. Those transposes also split the channels, so the channel count has to be at least `sp1` and at least `sp2`. The spectral weights are sharded over the same modes. `--tp=1`, because tensor parallelism applies only to the pointwise channel MLP. [`networks/fno_mp.py`](../../networks/fno_mp.py) is the model.

## No split

`short_mp` is the size matched to the ViT's `short_mp`: `embed_dim: 260`, `depth: 4`, modes 32×32, 555.1M real parameters. `downsample: 2` puts the blocks on a 2048×2048 latent grid. Local batch 1. `--tp=1 --sp1=1 --sp2=1` leaves the whole field on every GPU. Profiling stays off.

```bash
sbatch -N 1 --gpus-per-node 4 --cpus-per-task 48 \
    ./submit_mp.sh --yaml_config ./config/FNO.yaml --config short_mp --run_num fno_oom \
    --tp=1 --sp1=1 --sp2=1
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/1MP/fno_oom/out.log`.

Submit this job.

## Spatial split

`sp1=2` and `sp2=2` shard height and width inside each node, so the pencil-FFT all-to-alls stay on NVLink. Each rank's latent tile is 1024×1024. Two nodes leave `dp=4`, so the global batch of 4 keeps the local batch at 1. `--tp=1`.

```bash
ENABLE_PROFILING=1 PROFILE_OUTPUT=fno_mp \
sbatch -N 2 ./submit_mp.sh --yaml_config ./config/FNO.yaml --config short_mp --run_num fno_mp \
    --tp=1 --sp1=2 --sp2=2
```

Log: `$HOME/sc26-dl-tutorial/logs/short_mp/4MP/fno_mp/out.log`. Profiles: `fno_mp.0.nsys-rep` through `fno_mp.15.nsys-rep`.

Submit this job.
