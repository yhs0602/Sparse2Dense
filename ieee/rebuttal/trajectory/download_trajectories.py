import os
from datetime import datetime

import pandas as pd
import wandb
from tqdm import tqdm
from wandb.apis.public import Run, Runs

# from wandb_envs import WANDB_ENTITY, WANDB_PROJECT


current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)


def download_wandb_file(run: Run, download_dir):
    run_id = run.id
    run_group = run.group

    group_dir = os.path.join(download_dir, run_group)
    os.makedirs(group_dir, exist_ok=True)

    file_path = os.path.join(group_dir, f"{run_id}.csv")
    print(f"Downloading {run_id} to {file_path}")

    if not os.path.exists(file_path):
        scan_history = run.scan_history(keys=["episode", "episode/positions"])
        data = [
            [row.get(column) for column in ["_step", "episode", "episode/positions"]]
            for row in scan_history
        ]
        df = pd.DataFrame(data, columns=["_step", "episode", "episode/positions"])
        df.to_csv(file_path, compression="gzip")
        # history = run.history(keys=["episode", "episode/positions"], samples=500)
        # history.to_csv(file_path)
    else:
        print(f"File {file_path} already exists")
        return


def main():
    # Get all the runs from groups
    api = wandb.Api(timeout=120)
    WANDB_ENTITY = "jourhyang123"
    WANDB_PROJECT = "nature-journal-room_experiments"
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    group_runs = [run for run in runs if run.group.startswith("v31-room-")]
    print(f"Found {len(group_runs)} runs")

    with open("run_lists.csv", "w") as f:
        for run in group_runs:
            f.write(f"{run.id}, {run.group}\n")
    for run in tqdm(group_runs):
        download_wandb_file(run, "./all_run_trajectories")


normal_predicate = lambda run: (
    run.group.startswith("v33-room-")
    and datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    <= datetime(2024, 11, 27)
    and datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ") >= datetime(2024, 11, 1)
)

normal_s2d_predicate = lambda run: (
    normal_predicate(run) and "from_sparse" in run.group and "dense-False" in run.group
)

normal_s2d_begin_predicate = lambda run: (
    normal_predicate(run)
    and run.group.endswith("from_")
    and "sparse-False" in run.group
)

icm_predicate = lambda run: (run.group.startswith("v35-room-"))

icm_1_predicate = lambda run: (run.group.startswith("v34-room-"))

entropy_predicate = lambda run: (
    run.group.startswith("v33-room-")
    and datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    >= datetime(datetime.now().year, 11, 30)
)


def main241110():
    # Get all the runs from groups
    api = wandb.Api(timeout=120)
    WANDB_ENTITY = "jourhyang123"
    WANDB_PROJECT = "nature-journal-room_experiments"
    runs: Runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    group_runs = [run for run in runs if normal_predicate(run)]
    print(f"Found {len(group_runs)} runs")
    with open(
        f"{current_canonical_directory}/run_lists250715_trajectories.csv", "w"
    ) as f:
        for run in group_runs:
            f.write(f"{run.id}, {run.group}\n")
    for run in tqdm(group_runs):
        download_wandb_file(
            run, f"{current_canonical_directory}/all_run_trajectories250715-3"
        )


if __name__ == "__main__":
    main241110()
