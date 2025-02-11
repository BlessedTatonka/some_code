import os
import re
import glob
import argparse
import pandas as pd
from collections import defaultdict

def maybe_update_best(results_dict, task,
                      new_score, new_lr, new_n_epochs, new_wd, new_bsz, new_scheduler):
    """
    If `new_score` is better than what's stored (or ties but fewer epochs),
    update the entry in results_dict[task].
    """
    if task not in results_dict:
        results_dict[task] = (new_score, new_lr, new_n_epochs, new_wd, new_bsz, new_scheduler)
        return
    
    old_score, old_lr, old_n_epochs, old_wd, old_bsz, old_scheduler = results_dict[task]

    # Compare float values for the metric
    if new_score > old_score:
        results_dict[task] = (new_score, new_lr, new_n_epochs, new_wd, new_bsz, new_scheduler)
    elif new_score == old_score:
        # Compare integer epochs if the metric is tied
        if int(new_n_epochs) < int(old_n_epochs):
            results_dict[task] = (new_score, new_lr, new_n_epochs, new_wd, new_bsz, new_scheduler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_dir", 
        default="results_deepvk_moderna-small_stage_3__50B_tok", 
        help="Path to the top-level directory that contains subdirectories for each task/param setting."
    )
    args = parser.parse_args()

    # Regex pattern to parse subdirectory names like:
    #   danetqa_ft_lr=2e-05_n_epochs=2_wd=0.01_bsz=8_scheduler=cosine
    dir_pattern = re.compile(
        r"^(?P<task>[^_]+)"
        r"_ft_lr=(?P<lr>[^_]+)"
        r"_n_epochs=(?P<n_epochs>[^_]+)"
        r"_wd=(?P<wd>[^_]+)"
        r"_bsz=(?P<bsz>[^_]+)"
        r"_scheduler=(?P<scheduler>[^\.]+)$"
    )

    # Dictionaries to store best metrics
    best_any_row = {}
    best_last_row = {}

    # Dictionary to track the number of CSVs processed per task
    run_count_dict = defaultdict(int)

    # List all subdirectories in the results_dir
    subdirs = os.listdir(args.results_dir)
    if not subdirs:
        print(f"No subdirectories found in {args.results_dir}")
        return

    # Iterate over each subdirectory, parse out the params, gather CSVs
    for subdir_name in subdirs:
        full_subdir_path = os.path.join(args.results_dir, subdir_name)

        # We only care about actual directories
        if not os.path.isdir(full_subdir_path):
            continue

        # Try to match the pattern for the directory name
        match = dir_pattern.match(subdir_name)
        if not match:
            # If the directory doesn't match the naming pattern, skip
            continue

        # Extract parameters from the folder name
        task      = match.group("task")
        lr        = match.group("lr")
        n_epochs  = match.group("n_epochs")
        wd        = match.group("wd")
        bsz       = match.group("bsz")
        scheduler = match.group("scheduler")

        # Gather all CSV files in this subdir
        csv_files = glob.glob(os.path.join(full_subdir_path, "*.csv"))
        if not csv_files:
            print(f"No CSV files in subdir: {full_subdir_path}")
            continue

        # In your case, we've decided to look for "eval_eval_global_metric"
        metric_col = "eval_eval_global_metric"

        # Process each CSV file in the subdirectory
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if metric_col not in df.columns:
                # If the CSV doesn't have the column, skip it
                continue

            # We have a valid CSV with the metric, so increment run_count
            run_count_dict[task] += 1

            # Find best row's metric
            best_idx = df[metric_col].idxmax()
            best_val = df.loc[best_idx, metric_col]

            # Find last row's metric
            last_idx = df.index[-1]
            last_val = df.loc[last_idx, metric_col]

            # Update best-any-row
            maybe_update_best(
                best_any_row, task,
                best_val, lr, n_epochs, wd, bsz, scheduler
            )

            # Update best-last-row
            maybe_update_best(
                best_last_row, task,
                last_val, lr, n_epochs, wd, bsz, scheduler
            )

    # Build the final tables
    tasks = sorted(set(best_any_row.keys()) | set(best_last_row.keys()))

    # For "best (any row)"
    any_dict = {}
    for t in tasks:
        if t in best_any_row:
            score, lr, n_epochs, wd, bsz, scheduler = best_any_row[t]
            num_runs = run_count_dict[t]
            any_dict[t] = [
                f"{score:.4f}",  # Best Score (any row)
                lr,
                n_epochs,
                wd,
                bsz,
                scheduler,
                str(num_runs),
            ]
        else:
            # If a task has no best_any_row entry, it means no CSV matched
            # the required metric column for "any row"
            any_dict[t] = ["-", "-", "-", "-", "-", "-", "0"]

    # For "best (last row)"
    last_dict = {}
    for t in tasks:
        if t in best_last_row:
            score, lr, n_epochs, wd, bsz, scheduler = best_last_row[t]
            num_runs = run_count_dict[t]
            last_dict[t] = [
                f"{score:.4f}",  # Best Score (last row)
                lr,
                n_epochs,
                wd,
                bsz,
                scheduler,
                str(num_runs),
            ]
        else:
            last_dict[t] = ["-", "-", "-", "-", "-", "-", "0"]

    # Create DataFrames. Notice we now add an extra row for "num_runs".
    df_any = pd.DataFrame(
        any_dict,
        index=[
            "Best Score (any row)",
            "lr",
            "n_epochs",
            "wd",
            "bsz",
            "scheduler",
            "num_runs",
        ]
    )
    df_last = pd.DataFrame(
        last_dict,
        index=[
            "Best Score (last row)",
            "lr",
            "n_epochs",
            "wd",
            "bsz",
            "scheduler",
            "num_runs",
        ]
    )

    print("\n=== Absolute Best Metric (any row) ===")
    print(df_any)

    print("\n=== Best Metric (last row) ===")
    print(df_last)

if __name__ == "__main__":
    main()
