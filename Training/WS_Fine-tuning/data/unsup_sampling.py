import argparse
import json
import random
import math
from tqdm import tqdm
from datasets import load_dataset
from razdel import sentenize
import multiprocessing


def random_split(example, min_block_length=2):
    text = example["text"]
    sents = [sent.text for sent in sentenize(text)]
    num_sents = len(sents)

    if num_sents < 2 * min_block_length:
        return None

    # Chosing the first random segment
    a_start = random.randint(0, num_sents - 2 * min_block_length)
    a_max_length = num_sents - a_start - min_block_length
    a_length = random.randint(min_block_length, a_max_length)
    anchor = " ".join(sents[a_start : a_start + a_length])

    # Chosing the second random segment, which does not intersect the first
    p_start = random.randint(a_start + a_length, num_sents - min_block_length)
    p_max_length = num_sents - p_start
    p_length = random.randint(min_block_length, p_max_length)
    positive = " ".join(sents[p_start : p_start + p_length])

    return {"anchor": anchor, "positive": positive}


def process_example(example_and_args):
    example, min_block_length = example_and_args
    return random_split(example, min_block_length)


def init_worker(seed):
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Generate anchor-positive pairs from cultura dataset")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--max_samples", type=int, required=True, help="Maximum number of samples to generate")
    parser.add_argument("--min_block_length", type=int, default=2, help="Minimum number of sentences per block")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of multiprocessing workers")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Number of examples to group into a batch before processing in parallel")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset in streaming mode
    dataset = load_dataset("deepvk/cultura_ru_edu", split="train", streaming=True)

    samples_collected = 0

    # Set up multiprocessing pool
    with multiprocessing.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(args.seed, )
    ) as pool, open(args.output, "w", encoding="utf-8") as f:

        progress_bar = tqdm(total=args.max_samples, desc="Generating pairs")

        # Read chunks of examples
        chunk = []
        for example in dataset:
            # If we've reached the max, stop reading
            if samples_collected >= args.max_samples:
                break

            chunk.append((example, args.min_block_length))

            # If chunk is full, process in parallel
            if len(chunk) >= args.chunk_size:
                # Map chunk to the worker pool
                results = pool.map(process_example, chunk)

                # Write out valid results
                for result in results:
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        samples_collected += 1
                        progress_bar.update(1)
                        if samples_collected >= args.max_samples:
                            break

                # Clear chunk
                chunk = []

        # Process any leftover examples in the final (smaller) chunk
        if samples_collected < args.max_samples and len(chunk) > 0:
            results = pool.map(process_example, chunk)
            for result in results:
                if result is not None:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    samples_collected += 1
                    progress_bar.update(1)
                    if samples_collected >= args.max_samples:
                        break

        progress_bar.close()


if __name__ == "__main__":
    main()
