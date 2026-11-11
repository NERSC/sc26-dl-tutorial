#!/bin/bash
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-node 4
#SBATCH --time=00:30:00
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out

DATADIR=${DATADIR:-/fsx/datasets}
LOGDIR=${LOGDIR:-${HOME}/sc26-dl-tutorial/logs}
IMAGE=${IMAGE:-/scratch/containers/sc26-pytorch.sqsh}
mkdir -p ${LOGDIR}
args=("$@")

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --cuda-graph-trace=node --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=/logs/${PROFILE_OUTPUT:-"profile"}.%q{SLURM_PROCID}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

# if cuda graphs, use train_mp_graphs.py
set -x
srun -u --cpu-bind=none \
    --container-image="${IMAGE}" \
    --container-mounts="${DATADIR}:/data,${LOGDIR}:/logs" \
    --container-workdir="${SLURM_SUBMIT_DIR:-${PWD}}" \
    bash -c '
    source export_DDP_vars.sh
    export OMP_NUM_THREADS=1
    '"${PROFILE_CMD}"' python train_mp.py "$@"
    ' bash "${args[@]}"
