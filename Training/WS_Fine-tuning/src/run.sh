train_data="./training_data"
eval_data=""

# if we have 
max_steps=40_000
per_device_train_batch_size=16384
num_gpus=1

# mini_batch_size is the maximum amount of samples one GPU is capable of.
# https://github.com/UKPLab/sentence-transformers/issues/3207

model_args="\
    --model_name_or_path deepvk/RuModernBERT-small-retromae \
    --mini_batch_size 64 \
"

# If instructions path is specified, instructions will be applied

data_args="
    --train_data $train_data \
    --max_seq_length 8192 \
"

training_args="
    --output_dir output \
    --learning_rate 3e-4 \
    --max_steps $max_steps \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1_024 \
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

cmd="torchrun --nproc_per_node $num_gpus \
    -m run \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
