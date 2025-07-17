# Group by: algorithm
import glob
import os
import numpy as np
import pandas as pd


def render_trajectories(trajectory_dir: str):
    pass


def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    trajectories_dir = os.path.join(current_path, "all_run_trajectories250715")
    result_path = os.path.join(current_path, "sampled_trajectories")
    os.makedirs(result_path, exist_ok=True)
    for file in os.listdir(trajectories_dir):
        full_dir_path = os.path.join(trajectories_dir, file)
        if not os.path.isdir(full_dir_path):
            continue
        # parse the directory name
        seed = file.split("seed")[1].split("-")[0]
        start_algo = file.split("from_")[1]
        end_algo = file.split("v1-")[1].split("-")[0]
        print(f"Start algo: {start_algo}, End algo: {end_algo}, Seed: {seed}")
        # Sample 10 trajectories from each algo and save to another file
        matches = glob.glob(os.path.join(full_dir_path, "*.csv.gz"))
        if len(matches) == 0:
            print(f"Warning: No matches found in {full_dir_path}")
            continue
        csv_gz_path = matches[0]
        print(f"CSV GZ Path: {csv_gz_path}")

        # Read the csv.gz file and sample trajectories
        # with steps near 1000000, 2000000, 3000000, 4000000, ... 10000000
        df = pd.read_csv(csv_gz_path)
        global_steps = df["global_step"].values
        for target in range(0, 10000000, 1000000):
            result_rows = []
            idx = np.searchsorted(global_steps, target)
            start = max(0, idx - 15)
            end = min(len(global_steps), idx + 15)
            result_rows.append(df.iloc[start:end])
            result_df = pd.concat(result_rows, ignore_index=True)
            result_df.to_csv(
                os.path.join(
                    result_path,
                    f"{start_algo}_{end_algo}_{seed}_{target}.csv.gz",
                ),
                index=False,
                compression="gzip",
            )
            print(f"Saved {len(result_df)} trajectories to {target}")


# TODO: Render trajectories of;
# Algo     1000000 2000000 3000000 4000000 ... 10000000
# Sparse
# Dense
# S2D
# D2S

if __name__ == "__main__":
    main()
