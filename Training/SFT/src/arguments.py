import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:
    train_data: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrain data"}
    )
    eval_data: Optional[str] = field(
        default=None, metadata={"help": "Path to evaluation data"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    instructions_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If specified, then instructions will be applied."
        },
    )
    russian_only: Optional[bool] = field(
        default=False,
    )
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='answerdotai/ModernBERT-base',
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    mini_batch_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": ""
        },
    )