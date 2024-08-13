import os

import pandas as pd
import wandb
from tqdm import tqdm
from wandb.apis.public import Run

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT


def download_wandb_file(run: Run, download_dir):
    run_id = run.id
    run_group = run.group

    is_room = "room" in run_group

    group_dir = os.path.join(download_dir, run_group)
    os.makedirs(group_dir, exist_ok=True)

    file_path = os.path.join(group_dir, f"{run_id}.csv")
    print(f"Downloading {run_id} to {file_path}")

    if not os.path.exists(file_path):
        history = run.scan_history()
        # Metrics needed:
        if is_room:
            columns = [
                "step",
                "global_step",
                "episode",
                "episode/length",
                "0/success_rate",
                "1/success_rate",
                "2/success_rate",
                "3/success_rate",
                "0/success_count",
                "1/success_count",
                "2/success_count",
                "3/success_count",
                "episode/spawn_idx",
            ]
        else:
            columns = [
                "step",
                "global_step",
                "episode",
                "eval_episode",
                "episode/length",
                "episode/reward",
                "eval_episode/length",
                "eval_episode/reward",
                "goal_idx",
                "episode/goal_idx",
                "eval_episode/goal_idx",
                "eval_goal_idx",
            ]
        data = [[row.get(column) for column in columns] for row in history]
        df = pd.DataFrame(columns=columns, data=data)
        df.to_csv(file_path)
    else:
        print(f"File {file_path} already exists")
        return


def main():
    # Get all the runs from groups

    api = wandb.Api(timeout=120)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    group_runs = [
        run
        for run in runs
        if run.group.startswith("v31-room-") or run.group.startswith("v30-crossw2-")
    ]
    print(f"Found {len(group_runs)} runs")

    with open("run_lists.csv", "w") as f:
        for run in group_runs:
            f.write(f"{run.id}, {run.group}\n")
    for run in tqdm(group_runs):
        download_wandb_file(run, "./all_run_data")


if __name__ == "__main__":
    main()
