#!/bin/bash

DISTRIBUTED_ARGS="
    --nproc_per_node 7 \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


torchrun \
    --nproc_per_node="7" \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/open_r1/grpo_vllm.py \
    --deepspeed local_scripts/zero3_offload.json \
    --output_dir results \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name /path/to/dataset \
    --max_prompt_length 2048 \
    --num_generations 8 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 1000000 \
    --save_total_limit 30 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-7B-GRPO-8k \
    >> train.log 2>&1
