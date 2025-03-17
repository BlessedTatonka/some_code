train_data="/fast_nfs/spirin/weakly_sft"
eval_data="/fast_nfs/spirin/weakly_sft/validation"

# if we have 
max_steps=40_000
per_device_train_batch_size=2048  # Global: 2048 * 8 = 16384
num_gpus=8

# mini_batch_size is the maximum amount of samples one GPU is capable of.
# https://github.com/UKPLab/sentence-transformers/issues/3207

# model_args="\
#     --model_name_or_path deepvk/RuModernBERT-small-retromae \
#     --mini_batch_size 64 \
# "
model_args="\
    --model_name_or_path deepvk/RuModernBERT-small-retromae \
    --mini_batch_size 16 \
 "


# If instructions path is specified, instructions will be applied

data_args="
    --train_data $train_data \
    --eval_data $eval_data \
    --instructions_path instructions.yaml \
    --max_seq_length 8192 \
    --mixture_path mixture.yaml \
"

training_args="
    --output_dir /data/modern_bert/weakly_sft_small \
    --learning_rate 3e-4 \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 128 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --eval_strategy steps \
    --eval_steps 2_000 \
    --save_strategy steps \
    --save_steps 2_000 \
    --logging_steps 10 \
    --report_to wandb
"
# --ddp_find_unused_parameters True

cmd="torchrun --nproc_per_node $num_gpus \
    -m run \
    $model_args \
    $data_args \
    $training_args \
"

export TOKENIZERS_PARALLELISM=false
export WANDB_ENTITY=rumodernberta
export WANDB_PROJECT=weakly_sft

echo $cmd
eval $cmd
