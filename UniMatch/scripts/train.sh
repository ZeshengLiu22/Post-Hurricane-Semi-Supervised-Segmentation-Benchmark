#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these arguments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco', 'rescuenet', 'floodnet']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...]. Please check directory './splits/$dataset' for concrete splits
dataset=${DATASET:-rescuenet}
method=${METHOD:-unimatch}
exp=${EXP:-r101}
split=${SPLIT:-25}
num_gpus=${1:-${NUM_GPUS:-1}}
port=${2:-${PORT:-29500}}
visible_devices=${CUDA_VISIBLE_DEVICES:-0}

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=${SAVE_PATH:-exp/$dataset/$method/$exp/$split}

mkdir -p $save_path

if [[ "${CUDA_LAUNCH_BLOCKING:-0}" != "1" ]]; then
    unset CUDA_LAUNCH_BLOCKING
fi

CUDA_VISIBLE_DEVICES=$visible_devices torchrun \
    --nproc_per_node=$num_gpus \
    --master_addr=localhost \
    --master_port=$port \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $port 2>&1 | tee $save_path/$now.log
