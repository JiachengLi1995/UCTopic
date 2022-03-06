#!/bin/bash

# Set how many GPUs to use
NUM_GPU=2
# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.launch --nproc_per_node $NUM_GPU pretrain.py \
    --model_name_or_path studio-ousia/luke-base \
    --train_file pretraining_data/wiki_contrastive_entity_1m.json \
    --output_dir result/uctopic_base_pretraining \
    --num_train_epochs 1 \
    --preprocessing_num_workers 1 \
    --dataloader_num_workers 2  \
    --per_device_train_batch_size 50 \
    --per_device_eval_batch_size 50 \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --evaluation_strategy steps \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --eval_steps 5000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
