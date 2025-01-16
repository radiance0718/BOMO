#/bin/bash

# $1: task
# $2: gpu number

python3 sample_condition.py \
    --model_config=configs/model_config2.yaml \
    --diffusion_config=configs/diffusion_config_ddim2.yaml \
    --task_config=configs/noise_speckle_config2.yaml \
    --gpu=6 \
    --save_dir=./results/cm_dps_50;
