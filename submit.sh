#!/bin/bash
#SBATCH --cpus-per-task 24
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind none
#SBATCH --time=00:30:00
#SBATCH -J vit-era5
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
    NSYS_ARGS="--trace=cuda,cublas,nvtx --sample=none --cpuctxsw=none --resolve-symbols=false --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=/logs/${PROFILE_OUTPUT:-"profile"}.%q{SLURM_PROCID}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

set -x
srun -u --cpu-bind=none \
    --container-image="${IMAGE}" \
    --container-mounts="${DATADIR}:/data,${LOGDIR}:/logs" \
    --container-workdir="${SLURM_SUBMIT_DIR:-${PWD}}" \
    bash -c '
    '"${PROFILE_CMD}"' python train.py "$@"
    ' bash "${args[@]}"
