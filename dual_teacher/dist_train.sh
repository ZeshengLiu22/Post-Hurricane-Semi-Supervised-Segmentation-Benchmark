#!/bin/bash

# Number of GPUs to use
NUM_GPUS=8

# Port for distributed training (adjust if needed)
PORT=1777

# Training script and arguments
TRAIN_SCRIPT="tools/train-flood.py"
CONFIG_BACKBONE="mit_b1"
SAVE_PATH="logs/floodnet_run"
AMP="true"

# Create log directory if it doesn't exist
mkdir -p logs

# Launch the distributed training with nohup
nohup torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${PORT} \
    ${TRAIN_SCRIPT} \
    --ddp \
    --backbone ${CONFIG_BACKBONE} \
    --amp ${AMP} \
    --save_path ${SAVE_PATH} \
    > logs/train_$(date +%m%d_%H%M%S).log 2>&1 &
