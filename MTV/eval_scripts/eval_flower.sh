#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0
python3 mtv_eval.py \
    --model_name qwen2.5_vl \
    --data_name flower \
    --train_path ./data/flower/flower_train.json \
    --val_path ./data/flower/flower_test.json \
    --num_example 30 \
    --num_shot 4 \
    --max_token 20 \
    --eval_num_shot 0 \
    --bernoullis_path ./storage/flower_mtv.pt \
    --activation_path ./storage/flower_mtv_activation.pt \
    --is_eval True \
    --result_folder ./ \
    --cur_mode interv \
    --experiment_name temp