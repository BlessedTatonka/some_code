#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
from functools import partial

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ----------------------
# 1. Define tasks
# ----------------------
russian_superglue_tasks = {
    "rcb": {
        "abbr": "RCB",
        "name": "Russian Commitment Bank",
        "metrics": "F1/Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 3,
    },
    "parus": {
        "abbr": "PARus",
        "name": "Choice of Plausible Alternatives for Russian language",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["premise", "choice1", "choice2", "question"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "muserc": {
        "abbr": "MuSeRC",
        "name": "Russian Multi-Sentence Reading Comprehension",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["paragraph", "question", "answer"],
        "target": "label",
        "metric_funcs": [f1_score],
        "n_labels": 2,
    },
    "terra": {
        "abbr": "TERRa",
        "name": "Textual Entailment Recognition for Russian",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["premise", "hypothesis"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "russe": {
        "abbr": "RUSSE",
        "name": "Russian WiC - RUSSE",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["word", "sentence1", "sentence2"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "rwsd": {
        "abbr": "RWSD",
        "name": "Russian Winograd Schema Challenge",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["text", "span1_text", "span2_text"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
    "danetqa": {
        "abbr": "DaNetQA",
        "name": "Russian DaNetQA",
        "metrics": "Accuracy",
        "dataset_names": {"train": "train", "valid": "validation", "test": "test"},
        "inputs": ["question", "passage"],
        "target": "label",
        "metric_funcs": [accuracy_score],
        "n_labels": 2,
    },
}


# ----------------------
# 2. Seed fix function
# ----------------------
def fix_seed(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    set_seed(random_seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ----------------------
# 3. Compute metrics
# ----------------------
def compute_metrics(eval_pred, task_metrics):
    predictions, labels = eval_pred
    metrics_d = {}
    for metric_func in task_metrics:
        metric_name = metric_func.__name__
        if metric_name in ["f1_score"]:
            # For MuSeRC: f1_score
            score = metric_func(np.argmax(predictions, axis=-1), labels, average="binary")
        else:
            # For Accuracy tasks
            score = metric_func(np.argmax(predictions, axis=-1), labels)
        metrics_d[metric_name] = score
    return metrics_d


# ----------------------
# 4. Tokenizer function
# ----------------------
def preprocess_function(examples, task_inputs, hf_tokenizer):
    """
    Joins task inputs with a [SEP] token, then tokenizes.
    """
    input_sequences = zip(*[examples[inp] for inp in task_inputs])
    texts = [hf_tokenizer.sep_token.join(parts) for parts in input_sequences]
    # Adjust truncation or max_length as needed
    tokenized = hf_tokenizer(texts, truncation=False)
    return tokenized


# ----------------------
# 5. Main eval script
# ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or huggingface.co model ID from which to load the finetuned model.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        choices=list(russian_superglue_tasks.keys()),
        required=True,
        help="Russian SuperGLUE task to evaluate. E.g. `rcb`, `parus`, etc.",
    )
    args = parser.parse_args()

    # Fix seed for reproducibility
    fix_seed(42)

    # Setup logger (optional, you can remove if desired)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    logging.info(f"Evaluating task '{args.task_name}' using model '{args.model_path}' ...")

    # 1) Load the task metadata
    task_meta = russian_superglue_tasks[args.task_name]
    task_metrics = task_meta["metric_funcs"]
    valid_ds_name = task_meta["dataset_names"]["valid"]

    # 2) Load dataset
    logging.info("Loading raw dataset from `RussianNLP/russian_super_glue` ...")
    raw_datasets = load_dataset("RussianNLP/russian_super_glue", args.task_name)

    # 3) Load tokenizer
    logging.info("Loading tokenizer from model path ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 4) Tokenize datasets
    logging.info("Tokenizing validation split ...")
    tokenized_datasets = raw_datasets.map(
        lambda batch: preprocess_function(batch, task_meta["inputs"], tokenizer), batched=True
    )

    # 5) Load model
    logging.info("Loading model for evaluation ...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    # 6) Prepare Trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        do_train=False,
        do_eval=True,
        logging_steps=50,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, task_metrics=task_metrics),
    )

    # 7) Run evaluation on the validation set
    eval_dataset = tokenized_datasets[valid_ds_name]
    logging.info("Starting evaluation ...")
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)

    # 8) Print out all metrics
    logging.info(f"Evaluation results for task='{args.task_name}' on '{valid_ds_name}' split:")
    for k, v in eval_results.items():
        print(f"{k}: {v:.4f}")

    # 9) Print the "main" (global) metric value
    # By default, we'll check if we have an "eval_accuracy_score" or "eval_f1_score" etc.
    # Typically, there's exactly one custom metric in `eval_results` besides `eval_loss`.
    custom_metrics = {
        key: val
        for key, val in eval_results.items()
        if key.startswith("eval_")
        and key not in ["eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"]
    }

    # If there's only one, we can treat it as the global metric:
    if len(custom_metrics) == 1:
        metric_name, metric_val = list(custom_metrics.items())[0]
        print(f"Global metric [{metric_name}]: {metric_val:.4f}")
    elif len(custom_metrics) > 1:
        # If there's more than one, pick the first or handle as needed
        logging.warning("Multiple metrics found. Printing them all:")
        for k, v in custom_metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        logging.warning("No custom global metric found (other than loss).")


if __name__ == "__main__":
    main()
