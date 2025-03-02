from datasets import load_dataset, Dataset
from functools import partial
import os


def add_instructions(example, query_prefix, passage_prefix):
    query = example['anchor']
    passage = example['positive']
    
    query = query_prefix + query
    passage = query_prefix + passage
    
    return {"query": query, "passage": passage}


def load_multiple_datasets(data_path):
    
    if os.path.isdir(data_path):
        datasets = []
    
    dataset_dict = {}
    
    for file in os.listdir(data_path):
        dataset = Dataset.load_from_disk(os.path.join(data_path, file))
        
        # Nomic/FlagEmbedding mapping style
        if "anchor" not in dataset.column_names:
            dataset = dataset.rename_columns({"query": "anchor", "document": "positive"})
        
        dataset = dataset.select_columns(["anchor", "positive"])        
        dataset_dict[file] = dataset
        
    return dataset_dict
            
        