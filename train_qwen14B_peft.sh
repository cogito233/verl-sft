#!/bin/bash

set -x

# 设置 GPU 数量
nproc_per_node=4

# 保存路径
save_path=$HOME/verl/experiments/wiki/qwen2.5-7b/

# 环境变量
export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/WikiRL/data/wiki_sft/train.parquet \
    data.val_files=$HOME/WikiRL/data/wiki_sft/test.parquet \
    data.prompt_key=question \
    data.response_key=output \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=5120 \
    optim.lr=1e-4 \
    model.partial_pretrain=Qwen/Qwen2.5-7B \
    model.lora_rank=8 \
    model.lora_alpha=16 \
    model.target_modules=[q_proj,v_proj] \
    trainer.default_local_dir=$save_path \
    trainer.default_hdfs_dir=$save_path \
    trainer.project_name=wiki-sft \
    trainer.experiment_name=wiki-sft-qwen2.5-7b-peft \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']
