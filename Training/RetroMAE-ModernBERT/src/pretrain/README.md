# RetroMAE for ModernBert

## Key points:
- This code uses *mean* pooling;
- *learning_rate* and *mlm_probability* are taken from original paper;
- deepseed config can be used, ex. ds_config.json.

```bash
torchrun --nproc_per_node 8 -m run \
    --pretrain_method {retromae or dupmae} \
    --model_name_or_path deepvk/moderna-small \
    --model_revision stage_3__50B_tok \
    --output_dir output \
    --train_data <folder with training datasets> \
    --eval_data <folder with evaluation datasets> \
    --learning_rate 1e-4 \
    --max_steps 100_000 \
    --eval_strategy steps \
    --eval_steps 10_000 \
    --save_strategy steps \
    --save_steps 10_000 \
    --dataloader_drop_last True \
    --max_seq_length 8192 \
    --logging_steps 10 \
    --dataloader_num_workers 16 \
    --report_to wandb \
    --warmup_ratio 0.02 \
    --weight_decay 0.001 \
    --lr_scheduler_type inverse_sqrt \
    --encoder_mlm_probability 0.3 \
    --decoder_mlm_probability 0.5 \
    --dataloader_drop_last True \
    --per_device_train_batch_size 32
```