# 
# https://github.com/UKPLab/sentence-transformers/issues/2831

train_data="./train_datasets"
eval_data = ""

num_train_epochs=2
per_device_train_batch_size=2048

num_gpus=1

model_args="\
    --model_name_or_path deepvk/moderna-small \
    --mini_batch_size 1024 \
"

# If instructions path is specified, instructions will be applied

data_args="
    --train_data $train_data \
    --max_seq_length 128 \
    --instructions_path instructions.yaml \
"

training_args="
    --output_dir output \
    --learning_rate 3e-4 \
    --max_steps 1_000 \
    --per_device_train_batch_size 4_096 \
    --per_device_eval_batch_size 1_024 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --max_grad_norm 1.0 \
    --eval_strategy steps \
    --eval_steps 40 \
    --save_strategy steps \
    --save_steps 40 \
    --save_total_limit 2 \
    --logging_steps 1 \
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