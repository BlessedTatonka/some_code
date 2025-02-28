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


TRAIN_DATASETS = [
    {
        "dataset_path": "Jarikoff/unsup_data__ai-bond__ru-alpaca-summ",
        "query_prefix": "find_passage: ",
        "passage_prefix": "find_query: ",
        "pretty_name": "alpaca"
    },
    {
        "dataset_path": "Jarikoff/unsup_data__andidu__paraphrase-ru-reviews",
        "query_prefix": "find_paraphrase: ",
        "passage_prefix": "find_paraphrase: ",
        "pretty_name": "ru-reviews"
    }
]

EVAL_DATASETS = None


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SentenceTransformerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
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
            print(prompts)
        
        
    training_args.prompts = prompts

    # Loss
    train_loss = losses.CachedMultipleNegativesSymmetricRankingLoss(
        model,
        scale=100, # opposite to temperature, which should be 0.01
        mini_batch_size=model_args.mini_batch_size
    )

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
    main()
