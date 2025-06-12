import os

import pandas as pd
import argparse


def merge_and_save(base_file, million_file, num_million):
    base_df = pd.read_csv(base_file, index_col=0)
    million_df = pd.read_csv(million_file, index_col=0)
    # 3000320 is the last timestep => next one = + 512 = 3,000,832
    # 0 -> 0, 3000320 -> 5,860, 5859/3 = 1953
    # 5860 = 1 + 3 * 1953
    # 0, 1, ... 1953 : 1954개
    # 1954, ..., 3907: 1954개
    # 3908 ..., 5860: 1953개
    # 0 ~ 999,936
    # 1,000,448 ~ 2,000,384
    # 2,000,896 ~ 3,000,320
    base_df["global_step"] = base_df["global_step"].fillna(method="ffill")
    million_df["global_step"] = million_df["global_step"].fillna(method="ffill")

    if num_million == 1:
        filtered_df = base_df[base_df["global_step"] <= 999_936]
    elif num_million == 2:
        filtered_df = base_df[base_df["global_step"] <= 2_000_384]
    elif num_million == 3:
        filtered_df = base_df[base_df["global_step"] <= 3_000_320]
    else:
        raise ValueError(f"Invalid num_million: {num_million}")

    # concat the million df to the base df
    # but you should update the all continuating cumulative values

    # Calculate offsets
    episode_offset = filtered_df["episode"].max() + 1
    global_step_offset = filtered_df["global_step"].max() + 512

    # Adjust B's values
    million_df["episode"] = million_df["episode"] + episode_offset
    million_df["global_step"] = million_df["global_step"] + global_step_offset

    # update success counts and rates
    for i in range(4):
        success_count_col = f"{i}/success_count"
        visited_count_col = f"{i}/visited_count"
        success_rate_col = f"{i}/success_rate"

        filtered_df[visited_count_col] = (
            filtered_df[success_count_col] / filtered_df[success_rate_col]
        )
        million_df[visited_count_col] = (
            million_df[success_count_col] / million_df[success_rate_col]
        )

        max_visited_count = filtered_df[visited_count_col].max()
        max_success_count = filtered_df[success_count_col].max()

        million_df[visited_count_col] = (
            million_df[visited_count_col] + max_visited_count
        )
        million_df[success_count_col] = (
            million_df[success_count_col] + max_success_count
        )

        million_df[success_rate_col] = (
            million_df[success_count_col] / million_df[visited_count_col]
        )

    # concat the million df to the base df
    df_combined = pd.concat([filtered_df, million_df], ignore_index=True).reset_index(
        drop=True
    )

    merged_path = os.path.join(
        os.path.dirname(os.path.dirname(base_file)),
        "merged",
        f"{os.path.basename(base_file).split('.')[0]}_merged_{num_million}M.csv",
    )
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    df_combined.to_csv(merged_path, index=False)


def main(is_pbim: bool):
    bed_dir = "ir_runs_pbim" if is_pbim else "ir_runs"
    for algo in os.listdir(bed_dir):
        algo_dir = f"{bed_dir}/{algo}"
        base_dir = f"{algo_dir}/base"
        million_dir = f"{algo_dir}/100만"
        million_dir2 = f"{algo_dir}/200만"
        million_dir3 = f"{algo_dir}/300만"

        seeds = set()

        # for each csv in base, get seed
        files = {
            "base": {},
            "1M": {},
            "2M": {},
            "3M": {},
        }
        for csv in os.listdir(base_dir):
            if not csv.endswith(".csv"):
                continue
            seed = csv.split("-")[0]
            files["base"][seed] = f"{base_dir}/{csv}"
            seeds.add(seed)
        for csv in os.listdir(million_dir):
            if not csv.endswith(".csv"):
                continue
            seed = csv.split("-")[0]
            files["1M"][seed] = f"{million_dir}/{csv}"
            seeds.add(seed)
        for csv in os.listdir(million_dir2):
            if not csv.endswith(".csv"):
                continue
            seed = csv.split("-")[0]
            files["2M"][seed] = f"{million_dir2}/{csv}"
            seeds.add(seed)
        for csv in os.listdir(million_dir3):
            if not csv.endswith(".csv"):
                continue
            seed = csv.split("-")[0]
            files["3M"][seed] = f"{million_dir3}/{csv}"
            seeds.add(seed)
        print(files)
        print(seeds)

        for seed in seeds:
            base_file = files["base"].get(seed, None)
            if not base_file:
                print(f"No base file for seed {seed}")
                continue
            million_file = files["1M"].get(seed, None)
            if not million_file:
                print(f"No 1M file for seed {seed}")
                continue
            million_file2 = files["2M"].get(seed, None)
            if not million_file2:
                print(f"No 2M file for seed {seed}")
                continue
            million_file3 = files["3M"].get(seed, None)
            if not million_file3:
                print(f"No 3M file for seed {seed}")
                continue
            print(f"{base_file} + {million_file}")
            print(f"{base_file} + {million_file2}")
            print(f"{base_file} + {million_file3}")
            # Merge and fill the values
            merge_and_save(base_file, million_file, 1)
            merge_and_save(base_file, million_file2, 2)
            merge_and_save(base_file, million_file3, 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbim", action="store_true")
    args = parser.parse_args()
    main(args.pbim)
