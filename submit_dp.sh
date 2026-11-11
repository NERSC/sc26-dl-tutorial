#!/bin/bash
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 48
#SBATCH --gpus-per-node 4
#SBATCH --time=00:30:00
#SBATCH -J vit-era5-dp
#SBATCH -o %x-%j.out

DATADIR=${DATADIR:-/fsx/datasets}
LOGDIR=${LOGDIR:-${HOME}/sc26-dl-tutorial/logs}
IMAGE=${IMAGE:-/scratch/containers/sc26-pytorch.sqsh}
mkdir -p ${LOGDIR}
args=("$@")

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

# Profiling
PROFILE_CMD=()
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=/logs/${PROFILE_OUTPUT:-"profile"}.%q{RANK}
    PROFILE_CMD=(nsys profile ${NSYS_ARGS} -o "${NSYS_OUTPUT}")
fi

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2000))
NPROC_PER_NODE=${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}

set -x
srun -u --ntasks-per-node=1 --cpu-bind=none \
    --container-image="${IMAGE}" \
    --container-mounts="${DATADIR}:/data,${LOGDIR}:/logs" \
    --container-workdir="${SLURM_SUBMIT_DIR:-${PWD}}" \
    torchrun \
        --nnodes="${SLURM_JOB_NUM_NODES}" \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --no-python \
        "${PROFILE_CMD[@]}" python train.py "${args[@]}"
