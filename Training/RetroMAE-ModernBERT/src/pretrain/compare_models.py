import os
import json
import argparse
from statistics import mean
from tabulate import tabulate

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare MTEB benchmark scores across multiple models."
    )
    parser.add_argument(
        "model_directories",
        nargs='+',
        help="Paths to model directories containing MTEB JSON result files (searches subdirectories)."
    )
    return parser.parse_args()

def process_json_file(filepath):
    """
    Processes a single JSON file to extract main_scores.
    """
    scores = []
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            test_scores = data.get("scores", {}).get("test", [])
            for entry in test_scores:
                if "main_score" in entry:
                    scores.append(entry["main_score"])
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    return scores

def collect_task_scores(model_dir):
    """
    Collects task scores for a single model by searching JSON files in all subdirectories.
    """
    task_scores = {}
    for root, _, files in os.walk(model_dir):
        for file in files:
            if file.endswith(".json"):
                # Skip any meta file
                if file == "model_meta.json":
                    continue
                task_name = file.replace(".json", "")
                filepath = os.path.join(root, file)
                scores = process_json_file(filepath)
                if scores:
                    task_scores[task_name] = scores
    return task_scores

def get_model_name(model_path):
    """
    Extract the model name from the provided path.
    If the path matches 'results/mixed_llm/no_model_name_available/no_revision_available/',
    we use 'mixed_llm'. Otherwise, we use the directory basename.
    """
    # Normalize path to avoid trailing slashes issues
    norm_path = os.path.normpath(model_path)
    parts = norm_path.split(os.sep)

    # Check for 'mixed_llm/no_model_name_available/no_revision_available' pattern
    # Adjust indices if your directory structure is always: results/mixed_llm/no_model_name_available/no_revision_available
    # Example: parts = ['results', 'mixed_llm', 'no_model_name_available', 'no_revision_available']
    # Then parts[-4] would be 'results', parts[-3] = 'mixed_llm'
    # We'll check for the last 3 directories being exactly no_model_name_available/no_revision_available
    if len(parts) >= 4 and parts[-3:] == ["no_model_name_available", "no_revision_available"]:
        # Use the directory just before these last three for the name
        return parts[-4] if len(parts) >= 4 else "unknown_model"

    # Default to last part of the path
    return os.path.basename(norm_path)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    model_directories = args.model_directories

    # Validate model directories and collect their scores
    all_models = {}
    all_tasks = set()

    for model_path in model_directories:
        if not os.path.isdir(model_path):
            print(f"Error: The provided path '{model_path}' is not a directory. Skipping.")
            continue
        # Get the more human-friendly model name
        model_name = get_model_name(model_path)
        print(f"Processing model: {model_name}")

        model_scores = collect_task_scores(model_path)
        all_models[model_name] = model_scores
        all_tasks.update(model_scores.keys())

    if not all_models:
        print("No valid model directories with JSON files found.")
        return

    # Sort tasks for consistent display
    all_tasks = sorted(all_tasks)

    # Use the actual model names as headers
    models = list(all_models.keys())
    headers = ["Task"] + models

    # Build the table rows, but only include tasks that have scores for ALL models
    table = []
    for task in all_tasks:
        scores_per_model = []
        skip_task = False
        for model in models:
            model_scores = all_models[model].get(task, [])
            if not model_scores:
                # If this model doesn't have scores for this task, skip this task altogether
                skip_task = True
                break
            else:
                scores_per_model.append(mean(model_scores))
        
        if not skip_task:
            # All models have scores for this task
            row = [task] + [f"{s:.4f}" for s in scores_per_model]
            table.append(row)

    if not table:
        print("No tasks have scores for all models.")
        return

    # Compute average per model over the included tasks
    avg_row = ["AVERAGE"]
    for model_idx in range(1, len(headers)):
        scores_for_model = []
        for row in table:
            val = row[model_idx]
            if val != "N/A":  # Shouldn't be "N/A" at this point since we filtered them out
                scores_for_model.append(float(val))
        if scores_for_model:
            avg_row.append(f"{mean(scores_for_model):.4f}")
        else:
            avg_row.append("N/A")

    table.append(avg_row)

    # Print the table
    print("\nMTEB Benchmark Task Scores Comparison:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()
