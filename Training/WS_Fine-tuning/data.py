from datasets import load_dataset, Dataset
from functools import partial
import os


def add_instructions(example, query_prefix, passage_prefix):
    query = example['anchor']
    passage = example['positive']
    
    query = query_prefix + query
    passage = query_prefix + passage
    
    return {"query": query, "passage": passage}


def load_multiple_datasets(data_path, use_instructions=False):
    
    if os.path.isdir(data_path):
        datasets = []
    
    dataset_dict = {}
    
    for file in os.listdir(data_path):
        # dataset_path = ds['dataset_path']
        # query_prefix = ds['query_prefix']
        # passage_prefix = ds['passage_prefix']
        # pretty_name = ds['pretty_name']
        
        dataset = Dataset.load_from_disk(os.path.join(data_path, file))
        
        # Nomic/FlagEmbedding mapping style
        if "anchor" not in dataset.column_names:
            dataset = dataset.rename_columns({"query": "anchor", "document": "positive"})
        
        dataset = dataset.select_columns(["anchor", "positive"])
        
        # if use_instructions:
        #     dataset = dataset.map(partial(add_instructions, query_prefix, passage_prefix))
        
        dataset_dict[file] = dataset
        
    return dataset_dict
            
        