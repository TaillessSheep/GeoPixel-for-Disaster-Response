#!/bin/bash
#SBATCH --job-name=geopixel_finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
# Create Task: sbatch --account=mscaisuperpod --partition=normal --gres=gpu:2 --time=10:00:00 scripts/finetune_bg_template.sh
# Lookup Task: squeue -u $USER
# Cancel Task: scancel <JOBID>
# Task Logs: tail -n 100 logs/geopixel_finetune_322028.out
# Task Logs: tail -n 100 logs/geopixel_finetune_322028.err


source ~/.bashrc

PROJECT_ROOT=$(pwd)
CONTAINER_IMAGE="/home/$USER/containers/geopixel.sqsh"
CONTAINER_MOUNTS="${PROJECT_ROOT}:/workspace"

mkdir -p logs output

export CUDA_DEVICE_MAX_CONNECTIONS=1

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NODE_RANK=$SLURM_NODEID
export NNODES=$SLURM_NNODES

if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
fi

if [ "$GPUS_PER_NODE" -eq 0 ]; then
    GPUS_PER_NODE=4
fi

MODEL="MBZUAI/GeoPixel-7B-RES"
DATA="data.txt"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PYTHONWARNINGS="ignore" srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="/workspace" \
    --container-remap-root \
    torchrun $DISTRIBUTED_ARGS train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --resume_from_checkpoint False \
    --is_pretrained True \
    --given_num False \
    --bf16 True \
    --fix_vit True \
    --fix_sampler False \
    --use_lora True \
    --hd_num 1 \
    --output_dir output \
    --num_train_epochs 200 \
    --batch_size 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "./logs" \
    --report_to "tensorboard" \
    --max_length 16384 \
    --deepspeed ds_config_zero2.json \
    --gradient_checkpointing True