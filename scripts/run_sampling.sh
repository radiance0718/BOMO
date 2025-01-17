#/bin/bash

# $1: gpu number

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config_ddim.yaml \
    --task_config=configs/noise_speckle_config.yaml \
    --gpu=$1 \
    --save_dir=./results/BOMO;