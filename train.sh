#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export MODEL="internlm/internlm-xcomposer2d5-7b"
# export DATA="path of data"
export DATA="data.txt"

GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PYTHONWARNINGS="ignore" torchrun $DISTRIBUTED_ARGS train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --is_pretrained False \
    --given_num True \
    --bf16 True \
    --fix_vit True \
    --fix_sampler False \
    --use_lora True \
    --hd_num 9 \
    --output_dir output \
    --num_train_epochs 9\
    --batch_size 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --warmup_steps 150 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --logging_dir "./logs" \
    --report_to "tensorboard" \
    --max_length 16384 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True