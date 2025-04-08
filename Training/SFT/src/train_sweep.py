import logging
import os
from datetime import datetime

import wandb
import yaml
import transformers
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import is_main_process

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers, MultiDatasetBatchSamplers

from arguments import DataTrainingArguments, ModelArguments
from data import load_multiple_datasets
from evaluation import ir_evaluate, sts_evaluate

logger = logging.getLogger(__name__)


def train_sweep():
    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, :))
    sweep_config = wandb.config  # retrieve your sweep config first

    # Model args
    model_args = ModelArguments(
        model_name_or_path="/data/Models/USER_v2/weakly_sft/",
        mini_batch_size=8,
        temperature=sweep_config.temperature,
    )

    # Data args
    data_args = DataTrainingArguments(
        train_data="./training_data",
        eval_data="./validation_data",
        max_seq_length=8192,
        instructions_path="sft_instructions.yaml",
        russian_only=False,
    )

    # Training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir="output",
        learning_rate=sweep_config.learning_rate,
        num_train_epochs=1,
        per_device_train_batch_size=8128,
        per_device_eval_batch_size=128,
        warmup_ratio=0.05,
        weight_decay=sweep_config.weight_decay,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        report_to="wandb",
        save_total_limit=2,
        bf16=True,
        bf16_full_eval=True,
        dataloader_drop_last=True,
        load_best_model_at_end=True,
        metric_for_best_model="IR_cosine_ndcg@10",
    )
        
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )
    
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: SentenceTransformerTrainingArguments
    
    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    set_seed(training_args.seed)


    model = SentenceTransformer(
        model_args.model_name_or_path, 
        
    )
    if data_args.max_seq_length is not None:
        model.max_seq_length = data_args.max_seq_length

    # TRAINING DATA
    train_dataset_dict = load_multiple_datasets(data_args.train_data)
    
    eval_dataset_dict = None
    if data_args.eval_data is not None:
        eval_dataset_dict = load_multiple_datasets(data_args.eval_data)

    print("Train dataset dict:", train_dataset_dict)
    print("Eval dataset dict:", eval_dataset_dict)

    # EVALUATORS
    use_instructions = False
    if data_args.instructions_path is not None:
        use_instructions = True
    retrieval_evaluator = ir_evaluate(use_instructions=use_instructions)
    sts_evaluator = sts_evaluate() # Here we will use default instruction


    training_args.multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL
    training_args.batch_sampler = BatchSamplers.NO_DUPLICATES
    
    if use_instructions:
        with open(data_args.instructions_path, 'r') as file:
            prompts = yaml.safe_load(file)
            #print(prompts)
        training_args.prompts = prompts

    # We'll use gist embed loss with ubge-m3 model as a teacher
    guide_model = SentenceTransformer('/data/Models/USER_v2/weakly_sft/')

    # Loss
    base_loss = losses.CachedGISTEmbedLoss(
        model,
        temperature=model_args.temperature,
        guide=guide_model,
        mini_batch_size=model_args.mini_batch_size
    )
    
    train_loss = losses.MatryoshkaLoss(model, base_loss, [384, 256, 128, 64, 32])

    # Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_dict,
        eval_dataset=eval_dataset_dict,
        loss=train_loss,
        evaluator=[sts_evaluator, retrieval_evaluator],
    )

    # Start training
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    
    
    wandb.finish()
    logger.info("Training complete.")

if __name__ == "__main__":
    wandb.init()
    train_sweep()
