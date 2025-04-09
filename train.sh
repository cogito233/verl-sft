#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/WikiRL/data/wiki_sft/train.parquet \
    data.val_files=$HOME/WikiRL/data/wiki_sft/test.parquet \
    data.prompt_key=question \
    data.response_key=output \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=5120 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B \
    trainer.default_hdfs_dir=$HOME/verl/experiments/wiki/qwen2.5-1.5b/ \
    trainer.project_name=wiki-sft \
    trainer.experiment_name=wiki-sft-qwen2.5-1.5b \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb']

