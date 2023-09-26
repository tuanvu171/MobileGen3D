#!/bin/bash

export ROOT_DIR=../../../dataset/nerf_data
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name $scene  --dataset_name colmap\
    --num_epochs 25 --scale 1 --downsample 1  --lr 2e-2
