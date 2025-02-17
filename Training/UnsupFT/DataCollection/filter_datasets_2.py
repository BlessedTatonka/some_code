import datasets
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

model = SentenceTransformer('sergeyzh/rubert-tiny-turbo')

def calc_similarities(
        dataset_path, 
        anchor_col, 
        pos_col,
        name=None,
        low_sim=0.8,
        high_sim=0.9,
        batch_size=64):
    
    print("Start processing dataset:", dataset_path)
    # Load dataset
    if dataset_path.endswith('.jsonl'):
        dataset = datasets.Dataset.from_json(dataset_path)
    else:
        try:
            dataset = datasets.Dataset.load_from_disk(dataset_path)
        except:
            dataset = datasets.load_dataset(dataset_path, name=name)
    dataset = dataset['train']
    print(f"Dataset {dataset_path} length before filtration:", len(dataset))
        
    # Rename columns for consistency
    dataset = dataset.rename_columns({anchor_col: 'anchor', pos_col: 'positive'})
    
    # Filter dataset for samples that are within the allowed length
    def filter_length(sample):
        return (4 <= len(sample['anchor'])) and (4 <= len(sample['positive']))
    
    dataset = dataset.filter(filter_length)
    
    # Create a list of pairs from the filtered dataset
    pairs = [(sample['anchor'][:8192 * 4], sample['positive'][:8192 * 4]) for sample in dataset]

    similarities = []

    for start_idx in tqdm(range(0, len(pairs), batch_size)):
        batch_pairs = pairs[start_idx : start_idx + batch_size]
        
        # Prepare texts for encoding
        to_encode = []
        for (desc, ans) in batch_pairs:
            to_encode.append(desc)
            to_encode.append(ans)
        
        embeddings = model.encode(to_encode)
        
        for i, (que, ans) in enumerate(batch_pairs):
            anchor_emb = embeddings[2*i]
            positive_emb  = embeddings[2*i + 1]
            
            similarity = float(util.dot_score(anchor_emb, positive_emb))
            similarities.append(similarity)
    
    # Add similarity scores as a new column to the filtered dataset
    dataset = dataset.add_column('similarity', similarities)
    
    # Add task name information
    task_name = dataset_path.replace('/', '__')
    dataset = dataset.map(lambda x: {"task_name": task_name})
    dataset = dataset.select_columns(['anchor', 'positive', 'similarity', 'task_name'])
    
    # Filter based on similarity thresholds
    dataset = dataset.filter(lambda x: low_sim <= x['similarity'] <= high_sim)
    
    print(f"Dataset {dataset_path} length after filtration:", len(dataset))
    
    # Push the dataset to the hub
    dataset.push_to_hub(f'unsup_data__{task_name}_{anchor_col}_{pos_col}', private=True)
    
    return dataset

datasets_to_process = [
    {
        'dataset_path': 'data-silence/gazeta.ru-merged',
        'anchor_col': 'title',
        'pos_col': 'news',
        'low_sim': 0.8,
        'high_sim': 0.95,
    },
    {
        'dataset_path': 'data-silence/gazeta.ru-merged',
        'anchor_col': 'resume',
        'pos_col': 'news',
        'low_sim': 0.8,
        'high_sim': 0.95,
    },
]

for dataset in datasets_to_process:
    print('Processing dataset:', dataset['dataset_path'])
    try:
        _ = calc_similarities(**dataset)
    except:
        continue
    print('Succesfully processed dataset:', dataset['dataset_path'])
