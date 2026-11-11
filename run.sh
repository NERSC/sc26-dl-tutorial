#!/bin/bash
# Single-GPU interactive run, inside an existing allocation.
# Everything after the script name is forwarded to train.py.
#
#   ENABLE_PROFILING=1 PROFILE_OUTPUT=vit_short \
#   ./run.sh --yaml_config ./config/ViT.yaml --config short --run_num short

DATADIR=${DATADIR:-/pscratch/sd/s/shas1693/data/sc26_tutorial_data}
LOGDIR=${LOGDIR:-${SCRATCH}/sc26-dl-tutorial/logs}
IMAGE=${IMAGE:-nersc/pytorch:26.07.01}
NNODES=${NNODES:-1}
NGPU=${NGPU:-1}

mkdir -p "${LOGDIR}"
args=("$@")

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(hostname)

if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-profile}.%q{SLURM_PROCID}
    PROFILE_CMD="nsys profile ${NSYS_ARGS} -o ${NSYS_OUTPUT}"
fi

set -x
srun -u -N "${NNODES}" --ntasks-per-node="${NGPU}" --gpus-per-node="${NGPU}" --cpus-per-task=32 \
    shifter --image="${IMAGE}" -V "${DATADIR}:/data" -V "${LOGDIR}:/logs" \
    --module=gpu,nccl-cu13-plugin \
    bash -c '
    source export_DDP_vars.sh
    unset NCCL_NET_PLUGIN
    export OMP_NUM_THREADS=1
    '"${PROFILE_CMD}"' python train.py "$@"
    ' bash "${args[@]}"
