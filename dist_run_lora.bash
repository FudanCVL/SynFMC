#!/bin/bash
set -x

RANDOM_PORT=$((49152 + RANDOM % 16384))

gpus='0,1,2,3'
num_gpus=4



CUDA_VISIBLE_DEVICES=$gpus torchrun \
--master_port $RANDOM_PORT \
--nnodes=1 \
--nproc_per_node=$num_gpus \
train_image_lora.py \
--config configs/lora.yaml \
--launcher pytorch