#!/bin/bash
DATADIR=/pscratch/sd/s/shas1693/data/sc26_tutorial_data
LOGDIR=${SCRATCH}/sc26-dl-tutorial/logs
IMAGE=nersc/pytorch:26.07.01
ENABLE_PROFILING=1
CONFIG=short_mp

mkdir -p "${LOGDIR}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)

# Profiling
if [ "${ENABLE_PROFILING}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi --capture-range-end=stop -f true --trace-fork-before-exec=true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}_${CONFIG}.%q{SLURM_PROCID}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

set -x
nnodes=4
ngpu=4
srun -u -N ${nnodes} --ntasks-per-node=${ngpu} --gpus-per-node=${ngpu} --cpus-per-task=32 \
    shifter --image=${IMAGE} -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    --module=gpu,nccl-cu13-plugin \
    bash -c "
    source export_DDP_vars.sh
    unset NCCL_NET_PLUGIN
    ${PROFILE_CMD} python train_mp.py \
        --config=${CONFIG} \
        --tp=1 \
        --sp1=4 \
        --sp2=1
    "
