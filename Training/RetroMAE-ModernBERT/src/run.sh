#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: run.sh <TRAIN_DIR> <EVAL_DIR> <SAVE_DIR>"
  exit
fi

TRAIN_DIR=$1
EVAL_DIR=$2
SAVE_DIR=$3

TOKENIZERS_PARALLELISM=false WANDB_ENTITY=rumodernberta WANDB_PROJECT=retromae torchrun --nproc_per_node 8 -m pretrain.run \
    --pretrain_method retromae \
    --model_name_or_path deepvk/RuModernBERT-small \
    --output_dir $SAVE_DIR \
    --train_data $TRAIN_DIR \
    --eval_data $EVAL_DIR \
    --learning_rate 1e-4 \
    --max_steps 10_000 \
    --eval_strategy steps \
    --eval_steps 1_000 \
    --save_strategy steps \
    --save_steps 1_000 \
    --dataloader_drop_last True \
    --max_seq_length 8192 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --report_to wandb \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --lr_scheduler_type inverse_sqrt \
    --encoder_mlm_probability 0.3 \
    --decoder_mlm_probability 0.5 \
    --dataloader_drop_last True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --bf16 True \
    --deepspeed pretrain/ds_config.json \
    --resume_from_checkpoint /data/modern_bert/retromae_small/checkpoint-9000
