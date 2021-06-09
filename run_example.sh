#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=2
# Allow multiple threads
export OMP_NUM_THREADS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
#CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node $NUM_GPU train.py \
#CUDA_VISIBLE_DEVICES=4 python train.py \
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node $NUM_GPU train.py \
    --model_name_or_path studio-ousia/luke-base \
    --train_file data/wiki_contrastive_entity_1m.json \
    --output_dir result/uctopic_base \
    --num_train_epochs 3 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 4  \
    --per_device_train_batch_size 50 \
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
