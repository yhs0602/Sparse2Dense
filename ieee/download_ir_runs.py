import os
import pandas as pd
import wandb

from tqdm import tqdm
import argparse
from wandb.apis.public import Run


class IRRun:
    def __init__(self, algo, timing, seed, run_id):
        self.algo = algo
        self.timing = timing
        self.seed = seed
        self.run_id = run_id

    def __repr__(self):
        return f"IRRun(algo={self.algo}, timing={self.timing}, seed={self.seed}, run_id={self.run_id})"


class IRRunGroup:
    def __init__(self, algo):
        self.algo = algo
        self.runs = {}

    def add_run(self, run):
        if "까지" in run.timing:
            timing = "base"
        else:
            timing = run.timing
        timing = timing.replace(" ", "")
        if timing not in self.runs:
            self.runs[timing] = {}
        self.runs[timing][run.seed] = run

    def get_base_run(self, seed):
        return self.runs["base"][seed]

    def get_full_runs(self):
        for seed, base_run in self.runs["base"].items():
            full_runs = []
            for timing in self.runs.keys():
                if timing == "base":
                    continue
                full_runs.append(self.runs[timing][seed])
            yield full_runs


def main(is_pbim: bool = False):
    # load latest_runs.csv
    if is_pbim:
        df = pd.read_csv("latest_runs_pbim.csv")
    else:
        df = pd.read_csv("latest_runs.csv")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()
    print(df.head())

    print("Columns:")
    print(df.columns.tolist())

    ir_runs = []

    for _, row in df.iterrows():
        algo = row["algo"]
        timing = row["체크포인트"]
        for seed in row.index:
            if seed in ["algo", "체크포인트"]:
                continue
            run_id = row[seed]
            ir_runs.append(IRRun(algo=algo, timing=timing, seed=seed, run_id=run_id))

    for ir_run in ir_runs:
        print(ir_run)

    group_by_algo_checkpoint = {}
    for ir_run in ir_runs:
        if ir_run.algo not in group_by_algo_checkpoint:
            group_by_algo_checkpoint[ir_run.algo] = {}
        if ir_run.timing not in group_by_algo_checkpoint[ir_run.algo]:
            group_by_algo_checkpoint[ir_run.algo][ir_run.timing] = []
        group_by_algo_checkpoint[ir_run.algo][ir_run.timing].append(ir_run)

    print(group_by_algo_checkpoint)

    api = wandb.Api(timeout=120)

    for algo, checkpoint_group in tqdm(
        group_by_algo_checkpoint.items(),
        total=len(group_by_algo_checkpoint),
        desc="Downloading runs",
        position=0,
    ):
        for checkpoint, ir_runs in tqdm(
            checkpoint_group.items(),
            total=len(checkpoint_group),
            desc=f"Downloading runs for {algo}",
            position=1,
        ):
            if "까지" in checkpoint:
                checkpoint = "base"
            os.makedirs(f"ir_runs/{algo}/{checkpoint}", exist_ok=True)
            for ir_run in tqdm(
                ir_runs,
                total=len(ir_runs),
                desc=f"Downloading runs for {algo} {checkpoint}",
                position=2,
            ):
                # save as {seed}-{run_id}.csv
                file_name = f"{ir_run.seed}-{ir_run.run_id}.csv"
                if is_pbim:
                    file_path = f"ir_runs_pbim/{algo}/{checkpoint}/{file_name}"
                else:
                    file_path = f"ir_runs/{algo}/{checkpoint}/{file_name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if not os.path.exists(file_path):
                    if pd.isna(ir_run.run_id) or not ir_run.run_id:
                        print(f"Run {ir_run.run_id} is nan")
                        continue
                    # download from wandb
                    run: Run = api.run(
                        f"jourhyang123/Sparse2Dense-room_experiments/{ir_run.run_id}"
                    )
                    if run.state != "finished":
                        print(f"Run {ir_run.run_id} is not finished")
                        continue
                    history = run.scan_history()
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
                        "scaled_mean_intrinsic_rewards",
                        "episode/reward",
                    ]
                    data = [[row.get(column) for column in columns] for row in history]
                    df = pd.DataFrame(columns=columns, data=data)
                    df.to_csv(file_path)
                    print(f"Downloaded {file_path}")
                else:
                    print(f"File {file_path} already exists")


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--pbim", action="store_true")
    args = argparse.parse_args()
    main(args.pbim)


# 이제 내가 wandb에서 해당 run들을 다운받은다음에 그 run에 대하여
# global step 대비 episode,
# group by = algo, checkpoint
# 이후 checkpoint의 머시기까지 이거, 나머지 100만 200만 300만을 합쳐서
#
