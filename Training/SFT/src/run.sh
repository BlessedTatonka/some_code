train_data="./training_data"
eval_data="./validation_data"

max_steps=3_000
per_device_train_batch_size=1024
num_gpus=8

# mini_batch_size is the maximum amount of samples one GPU is capable of.
# https://github.com/UKPLab/sentence-transformers/issues/3207

model_args="\
    --model_name_or_path <###> \
    --mini_batch_size 8 \
    --instructions_path sft_instructions.yaml
"

# If instructions path is specified, instructions will be applied

data_args="
    --train_data $train_data \
    --eval_data $eval_data \
    --max_seq_length 8192 \
    --russian_only false
"

training_args="
    --output_dir output \
    --learning_rate 3e-5 \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 256 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --logging_steps 10 \
    --report_to wandb \
    --save_total_limit 2 \
    --bf16 \
    --bf16_full_eval \
    --dataloader_drop_last \
    --load_best_model_at_end \
    --metric_for_best_model IR_cosine_ndcg@10 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m run \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
