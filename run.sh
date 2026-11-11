#!/bin/bash
# Single-GPU interactive run, inside an existing allocation (salloc).
# Everything after the script name is forwarded to train.py.
#
#   ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short \
#   ./run.sh --yaml_config ./config/ViT.yaml --config short --run_num short

DATADIR=${DATADIR:-/fsx/datasets}
LOGDIR=${LOGDIR:-${SCRATCH:-${HOME}}/sc26-dl-tutorial/logs}
IMAGE=${IMAGE:-/scratch/containers/sc26-pytorch.sqsh}
NNODES=${NNODES:-1}
NGPU=${NGPU:-1}
NCPU=${NCPU:-24}

mkdir -p "${LOGDIR}"
args=("$@")

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
# salloc on ParallelCluster leaves you on the head node, so take the first allocated node
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=/logs/${PROFILE_OUTPUT:-profile}.%q{SLURM_PROCID}
    PROFILE_CMD="nsys profile ${NSYS_ARGS} -o ${NSYS_OUTPUT}"
fi

set -x
srun -u -N "${NNODES}" --ntasks-per-node="${NGPU}" --gpus-per-node="${NGPU}" \
    --cpus-per-task="${NCPU}" --cpu-bind=none \
    --container-image="${IMAGE}" \
    --container-mounts="${DATADIR}:/data,${LOGDIR}:/logs" \
    --container-workdir="${PWD}" \
    bash -c '
    source export_DDP_vars.sh
    export OMP_NUM_THREADS=1
    '"${PROFILE_CMD}"' python train.py "$@"
    ' bash "${args[@]}"
