#!/usr/bin/env python
import subprocess
import itertools

def main():
    model_name = "deepvk/moderna-small"
    model_revision = "stage_3__50B_tok"
    
    # Define the hyperparameter grid
    tasks = ["rcb", "parus", "muserc", "terra", "russe", "rwsd", "danetqa"]
    learning_rates = [2e-5, 5e-5, 1e-4]
    weight_decays = [0, 1e-2, 1e-1]
    batch_sizes = [8, 16, 32, 64]
    num_epochs = [2, 3, 5, 10]
    lr_scheduler_type = ['cosine', 'inverse_sqrt'] 
    
    grid = itertools.product(
        tasks, 
        learning_rates, 
        weight_decays, 
        batch_sizes, 
        num_epochs,
        lr_scheduler_type
    )
    
    # Iterate over each combination and run train_no_wandb.py
    for (task, lr, wd, bsz, epochs, sch) in grid:
        cmd = [
            "python", "train.py",
            "--model_name", model_name,
            "--model_revision", model_revision, 
            "--task_name", task,
            "--learning_rate", str(lr),
            "--weight_decay", str(wd),
            "--batch_size", str(bsz),
            "--num_train_epochs", str(epochs),
            "--lr_scheduler_type", str(sch)
        ]

        print(f"\n=== Running: {cmd} ===")
        # If something happens during one task?
        # Probably should be removed
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            # Re-raise so the program stops on Ctrl+C
            raise
        except Exception:
            pass
        print(f"--- Completed run for Task={task}, LR={lr}, WD={wd}, BSZ={bsz}, SCHEDULER={sch}, EPOCHS={epochs} ---")

if __name__ == "__main__":
    main()