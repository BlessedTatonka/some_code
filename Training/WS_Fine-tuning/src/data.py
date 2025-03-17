import json
import os

import numpy as np
import torch
from datasets import Dataset, DatasetInfo
from loguru import logger
from streaming.base.format import reader_from_json
from streaming.base.spanner import Spanner


def add_instructions(example, anchor, positive):
    return {"anchor": anchor + example["anchor"], "positive": positive + example["positive"]}


def load_multiple_datasets(data_path, instructions=None):

    if os.path.isdir(data_path):
        datasets = []

    dataset_dict = {}

    for file in os.listdir(data_path):
        dataset = Dataset.load_from_disk(os.path.join(data_path, file))

        # Nomic/FlagEmbedding mapping style
        if "anchor" not in dataset.column_names:
            dataset = dataset.rename_columns({"query": "anchor", "document": "positive"})

        dataset = dataset.select_columns(["anchor", "positive"])

        if instructions is not None and file in instructions:
            dataset = dataset.map(lambda x: add_instructions(x, **instructions[file]))
        elif instructions is not None:
            logger.warning(f"No instructions for '{file}'")
            dataset = dataset.map(lambda x: add_instructions(x, instructions["default"], instructions["default"]))

        dataset_dict[file] = dataset

    return dataset_dict


class WeaklySFTDataset(torch.utils.data.Dataset):
    """
    A dataset class that can read data with raw mds-format (mosaic streaming-format without compression)
    from local. In comparison with `StreamingTextDataset` that also can read data with mds-format from local,
    this class is slimmer, more efficient, and does not contain redundant code required for streaming.
    """

    def __init__(self, data_dir: str, repeat: int = 1):
        super().__init__()
        self.repeat = repeat
        self.name = os.path.basename(data_dir)
        index_file_path = os.path.join(data_dir, "index.json")
        with open(index_file_path) as f:
            obj = json.load(f)
        self.shards = []
        for info in obj["shards"]:
            shard = reader_from_json(data_dir, split=None, obj=info)
            raw_filename = os.path.join(shard.dirname, shard.split, shard.raw_data.basename)
            assert os.path.isfile(raw_filename) or os.path.islink(
                raw_filename
            ), f"Raw file {raw_filename} does not exist"
            shard.validate(True)
            self.shards.append(shard)
        samples_per_shard = np.array([shard.samples for shard in self.shards], np.int64)
        self.len = samples_per_shard.sum()
        self.spanner = Spanner(samples_per_shard)

        self.anchor_prefix = ""
        self.positive_prefix = ""

        # logger.info(f"Initialize '{self.name}' dataset with {self.len} samples, w.r.t. repeat: {self.len * self.repeat}")

    @property
    def column_names(self):
        return ["anchor", "positive"]

    @property
    def info(self):
        return DatasetInfo()

    @property
    def split(self):
        return "train"

    @property
    def download_checksums(self):
        return None

    def apply_instructions(self, anchor: str, positive: str):
        self.anchor_prefix = anchor
        self.positive_prefix = positive

    def __getitem__(self, index: int) -> dict[str, str]:
        true_index = index // self.repeat
        assert true_index < self.len

        shard_id, shard_sample_id = self.spanner[true_index]
        shard = self.shards[shard_id]
        sample = shard[shard_sample_id]
        return add_instructions(sample, self.anchor_prefix, self.positive_prefix)

    def __len__(self):
        return self.len * self.repeat


def load_from_mixture_configuration(data_path: str, configuration: list[tuple[str, int]], instructions):
    datasets = {}
    for dataset, repeat in configuration:
        dataset = WeaklySFTDataset(os.path.join(data_path, dataset), repeat=repeat)
        if instructions is not None:
            if dataset.name in instructions:
                dataset.apply_instructions(**instructions[dataset.name])
            else:
                logger.warning(f"No instructions for '{dataset.name}'")
                dataset.apply_instructions(anchor=instructions["default"], positive=instructions["default"])
        datasets[dataset.name] = dataset
    return datasets


if __name__ == "__main__":
    from argparse import ArgumentParser
    from yaml import safe_load

    _arg_parser = ArgumentParser()
    _arg_parser.add_argument("dataset_dir")
    _arg_parser.add_argument("mixture_path")
    _arg_parser.add_argument("--instructions", default=None)
    _arg_parser.add_argument("--n-samples", default=3)

    _args = _arg_parser.parse_args()
    with open(_args.mixture_path, "r") as file:
        mixture_config = safe_load(file)

    instructions = None
    if _args.instructions is not None:
        with open(_args.instructions, "r") as file:
            instructions = safe_load(file)

    _dataset2repeat = [(k, v["repeat"]) for k, v in mixture_config.items()]
    _datasets = load_from_mixture_configuration(_args.dataset_dir, _dataset2repeat, instructions)
    print(f"Total size: {sum((len(it) for it in _datasets.values()))}")

    for k, v in _datasets.items():
        print(k)
        print(v[0])
        print("=" * 80)
