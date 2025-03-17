#!/usr/bin/env python

import argparse
import csv
import os
from datetime import datetime
from importlib import reload

# Encodechka Eval
from encodechka_eval import tasks
from encodechka_eval.bert_embedders import embed_bert_both, get_word_vectors_with_bert
from transformers import AutoModel, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on multiple Encodechka tasks.")
    parser.add_argument(
        "--model", type=str, required=True, help="Name or path of the model (e.g. cointegrated/rubert-tiny)."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks to run (STS, PI, NLI, SA, TI, IA, IC, ICX, NE1, NE2). "
        "If not specified, will run all tasks.",
    )
    args = parser.parse_args()

    # Set where Encodechka data is stored (or ensure the folder is correct for your setup)
    DATA_PATH_NAME = "ENCODECHKA_DATA_PATH"
    os.environ[DATA_PATH_NAME] = "encodechka_eval"  # Make sure you have the data in this folder or update accordingly

    # Reload tasks to ensure environment variable is recognized
    reload(tasks)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.cuda()

    # Task runner functions
    def run_stsb():
        task_ = tasks.STSBTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_pi():
        task_ = tasks.ParaphraserTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_nli():
        task_ = tasks.XnliTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_sa():
        task_ = tasks.SentimentTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_ti():
        task_ = tasks.ToxicityTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_ia():
        task_ = tasks.InappropriatenessTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_ic():
        task_ = tasks.IntentsTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_icx():
        task_ = tasks.IntentsXTask()
        return task_.eval(lambda x: embed_bert_both(x, model, tokenizer), args.model)

    def run_ne1():
        task_ = tasks.FactRuTask()
        return task_.eval(lambda words: get_word_vectors_with_bert(words, model=model, tokenizer=tokenizer), args.model)

    def run_ne2():
        task_ = tasks.RudrTask()
        return task_.eval(lambda words: get_word_vectors_with_bert(words, model=model, tokenizer=tokenizer), args.model)

    # Define a mapping from short task name to the runner function
    task_mapping = {
        "STS": run_stsb,
        "PI": run_pi,
        "NLI": run_nli,
        "SA": run_sa,
        "TI": run_ti,
        "IA": run_ia,
        "IC": run_ic,
        "ICX": run_icx,
        "NE1": run_ne1,
        "NE2": run_ne2,
    }

    # Default order if no tasks are specified
    default_order = ["STS", "PI", "NLI", "SA", "TI", "IA", "IC", "ICX", "NE1", "NE2"]

    # Parse the user-supplied --tasks
    if args.tasks:
        # e.g. --tasks STS,SA,TI
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip() in task_mapping]
    else:
        # If not specified, run all tasks
        task_list = default_order

    if not task_list:
        print("No valid tasks specified. Exiting.")
        return

    # Run the tasks and store results
    results = []
    for task_short_name in task_list:
        print(f"Running task: {task_short_name} ...")
        score_tuple = task_mapping[task_short_name]()
        # Each task returns something like: (0.344516..., {'macro_f1': 0.344516...})
        # We only take the first value, the "main score"
        main_score = score_tuple[0]
        results.append((task_short_name, main_score))

    # Print table of results
    print("\n======== Evaluation Results ========")
    print(f"{'Task':<5} | Score")
    print("-" * 26)
    total_score = 0.0
    for short_name, score in results:
        total_score += score
        print(f"{short_name:<5} | {score:.6f}")
    avg_score = total_score / len(results)
    print("-" * 26)
    print(f"{'AVG':<5} | {avg_score:.6f}")
    print("====================================\n")

    # Save scores to CSV file
    # Directory: results_encodechka/<model>/<date_time>/results.csv
    date_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results_encodechka", args.model, date_time_str)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Score"])
        for short_name, score in results:
            writer.writerow([short_name, f"{score:.6f}"])
        writer.writerow(["AVG", f"{avg_score:.6f}"])

    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
