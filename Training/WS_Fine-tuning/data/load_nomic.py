import os
import re
import json
import yaml
import gzip
import argparse

import fsspec
from streaming import MDSWriter
from tqdm import tqdm

MDS_COLS_TEXT = {
    "anchor": "str",
    "positive": "str"
}

def expand_shard_pattern(pattern: str):
    """
    Expand a brace range like:
      'shard-{00000..00005}.jsonl.gz'
    into a list:
      ['shard-00000.jsonl.gz', 'shard-00001.jsonl.gz', ... 'shard-00005.jsonl.gz'].

    If no braces are found, return [pattern] as-is.
    """
    match = re.search(r'{(\d+)\.\.(\d+)}', pattern)
    if not match:
        return [pattern]

    start, end = int(match.group(1)), int(match.group(2))
    prefix = pattern[:match.start()]
    suffix = pattern[match.end():]

    expanded = []
    for i in range(start, end + 1):
        expanded.append(f"{prefix}{i:05d}{suffix}")
    return expanded

def convert_dataset_to_mds(dataset_cfg, output_base_dir):
    dataset_name = dataset_cfg["name"]
    shard_pattern = dataset_cfg["bucket"]
    columns = dataset_cfg["objective"]["columns"]

    shard_paths = expand_shard_pattern(shard_pattern)

    dataset_output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    index_file = os.path.join(dataset_output_dir, "index.json")
    if os.path.isfile(index_file):
        print(f"[SKIP] Dataset '{dataset_name}' already converted (index.json found).")
        return

    print(f"==> Converting dataset '{dataset_name}'")
    print(f"    Shard pattern: {shard_pattern}")
    print(f"    # of expanded shards: {len(shard_paths)}")
    print(f"    Output MDS: {dataset_output_dir}")

    # fsspec args
    fs_open_kwargs = {}
    s3_endpoint_url = "https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/"
    fs_open_kwargs["client_kwargs"] = {"endpoint_url": s3_endpoint_url}

    # Create the MDSWriter.
    with MDSWriter(out=dataset_output_dir, columns=MDS_COLS_TEXT, compression="zstd") as mds_writer:
        total_written = 0
        for shard_url in shard_paths:
            print(f"  -> Reading shard: {shard_url}")

            with fsspec.open(shard_url, "rb", **fs_open_kwargs) as raw_file:
                # Decompress
                with gzip.open(raw_file, "rt", encoding="utf-8") as shard_file:
                    for line in tqdm(shard_file, desc=f"Shard {os.path.basename(shard_url)}"):
                        record = json.loads(line)

                        # Rename columns
                        record["anchor"] = record.pop(columns[0])
                        record["positive"] = record.pop(columns[1])

                        # Write the record
                        mds_writer.write(record)
                        total_written += 1

        print(f"    Done. Wrote {total_written} records for '{dataset_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Convert a YAML list of datasets to MDS format.")
    parser.add_argument("--config", required=True, help="Path to your .yaml config file.")
    parser.add_argument("--output", required=True, help="Directory to store the MDS outputs.")
    args = parser.parse_args()

    # .yaml with dataset specs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    for ds_cfg in config["datasets"]:
        convert_dataset_to_mds(ds_cfg, args.output)

    print("All done!")

if __name__ == "__main__":
    main()

"""
example:
python3 load_nomic.py --config contrastive_pretrain_multilingual_en_ru.yaml --output training_data
"""
