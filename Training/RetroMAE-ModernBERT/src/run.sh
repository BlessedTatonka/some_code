torchrun --nproc_per_node 1 -m pretrain.run \
    --pretrain_method retromae \
    --model_name_or_path deepvk/moderna-small \
    --model_revision stage_3__50B_tok \
    --output_dir output_dupmae \
    --train_data /hdd/TEXT_datasets/RetroMAE_data \
    --eval_data /hdd/TEXT_datasets/RetroMAE_data_eval \
    --eval_strategy steps \
    --eval_steps 2_000 \
    --learning_rate 8e-5 \
    --max_steps 10_000 \
    --per_device_train_batch_size 24 \
    --dataloader_drop_last True \
    --max_seq_length 256 \
    --logging_steps 10 \
    --dataloader_num_workers 8 \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 2_000 \
    --warmup_ratio 0.1 \
    --weight_decay 0.001 \
    --lr_scheduler_type inverse_sqrt \
    --encoder_mlm_probability 0.3 \
    --decoder_mlm_probability 0.5 \
    --dataloader_drop_last True
