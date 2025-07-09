#!/bin/bash

set -x
RANDOM_PORT=$((49152 + RANDOM % 16384))


gpus='1,2,3,4,5,6,7'
num_gpus=7



CUDA_VISIBLE_DEVICES=$gpus torchrun \
--master_port $RANDOM_PORT \
--nnodes=1 \
--nproc_per_node=$num_gpus \
train_cam_obj_ctrl.py \
--config configs/obj.yaml \
--launcher pytorch



