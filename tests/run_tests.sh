#!/bin/bash

image=nersc/pytorch:26.07.01
dp=1
tp=1
sp1=1
sp2=1
nodes=1
model=vit
backend=mp

# parse args
for arg in "$@"
do
    if [[ $arg == --model=* ]]; then
        model="${arg#*=}"
    elif [[ $arg == --backend=* ]]; then
        backend="${arg#*=}"
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

if [[ $model != vit && $model != swin && $model != fno && $model != convnext ]]; then
    echo "ERROR: --model must be vit, swin, fno, or convnext" >&2
    exit 2
fi

if [[ $backend == te ]]; then
    test_file=tests/test_all_te.py
    if [[ $model != vit ]]; then
        echo "ERROR: --backend=te only supports --model=vit" >&2
        exit 2
    fi
elif [[ $backend == mp ]]; then
    test_file=tests/test_all.py
else
    echo "ERROR: --backend must be mp or te" >&2
    exit 2
fi

ngpu_per_node=$(( (${tp} * ${sp1} * ${sp2} * ${dp})/$nodes ))
export MASTER_ADDR=$(hostname)
export OMP_NUM_THREADS=1
set -x
srun -l --nodes $nodes --ntasks-per-node $ngpu_per_node --gpus-per-node $ngpu_per_node -u shifter --image=$image --module=gpu,nccl-cu13-plugin \
    bash -c "
    source export_DDP_vars.sh
    unset NCCL_NET_PLUGIN
    export OMP_NUM_THREADS=1
    export NVIDIA_TF32_OVERRIDE=0
    export TP=${tp}
    export SP1=${sp1}
    export SP2=${sp2}
    export TEST_MODEL=${model}
    python -m pytest -s ${test_file}
    "
