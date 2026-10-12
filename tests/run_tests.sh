#!/bin/bash

image=nersc/pytorch:25.06.01
dp=1
tp=1
sp1=1
sp2=1
nodes=1
model=vit

# parse args
for arg in "$@"
do
    if [[ $arg == --model=* ]]; then
        model="${arg#*=}"
    elif [[ $arg == --tp=* ]]; then
        tp="${arg#*=}"
    elif [[ $arg == --sp1=* ]]; then
        sp1="${arg#*=}"
    elif [[ $arg == --sp2=* ]]; then
        sp2="${arg#*=}"
    elif [[ $arg == --dp=* ]]; then
        dp="${arg#*=}"
    elif [[ $arg == --nodes=* ]]; then
        nodes="${arg#*=}"
    fi
done

if [[ $model != vit && $model != swin && $model != fno ]]; then
    echo "ERROR: --model must be vit, swin, or fno" >&2
    exit 2
fi

ngpu_per_node=$(( (${tp} * ${sp1} * ${sp2} * ${dp})/$nodes ))
export MASTER_ADDR=$(hostname)
set -x
srun -l --nodes $nodes --ntasks-per-node $ngpu_per_node --gpus-per-node $ngpu_per_node -u shifter --image=$image --module=gpu,nccl-plugin \
    bash -c "
    source export_DDP_vars.sh
    export NVIDIA_TF32_OVERRIDE=0
    export TP=${tp}
    export SP1=${sp1}
    export SP2=${sp2}
    export TEST_MODEL=${model}
    python -m pytest -s tests/test_all.py
    "
