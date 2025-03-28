from datasets import load_dataset, Dataset
from functools import partial
import os


def add_instructions(example, query_prefix, passage_prefix):
    query = example['anchor']
    passage = example['positive']
    
    query = query_prefix + query
    passage = query_prefix + passage
    
    return {"query": query, "passage": passage}


def load_multiple_datasets(data_path, russian_only=False):
    
    dataset_dict = {}
    
    available_langs = None
    if russian_only:
        available_langs = ['RU']
    else:
        available_langs = os.listdir(data_path)
    
    for lang in available_langs:
        for file in os.listdir(os.path.join(data_path, lang)):
            dataset = Dataset.load_from_disk(os.path.join(data_path, lang, file))
            
            # currently taking only one hard neg
            if 'negative_1' in dataset.column_names:
                dataset = dataset.select_columns(["anchor", "positive", "negative_1"]) 
                       
            dataset_dict[file] = dataset
    
    return dataset_dict
            
        