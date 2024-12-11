import os
from collections import defaultdict
from typing import Tuple

import pandas as pd


current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)
base_dir = f"{current_canonical_directory}/../all_run_data241211-entropy/"


def extract_info(string) -> Tuple[int, bool]:
    parts = string.split("-")
    seed_part = parts[5]
    # get only integer part
    seed = int(seed_part.split("seed")[1])
    from_part = parts[6]
    is_after = len(from_part) > 5
    return (seed, is_after)


def iterate_and_group_by_seed():
    seed_groups = defaultdict(list)
    # Iterate through directories and parse information
    for dir in os.listdir(base_dir):
        seed, is_after = extract_info(dir)
        print(f"{dir} -> Seed: {seed}, is_after: {is_after}")
        # iterate csvs
        files = [
            f for f in os.listdir(os.path.join(base_dir, dir)) if f.endswith(".csv")
        ]
        seed_groups[seed].append({"path": dir, "is_after": is_after, "files": files})

    result = {}
    for seed in seed_groups:
        start_group = [group for group in seed_groups[seed] if not group["is_after"]][0]
        end_group = [group for group in seed_groups[seed] if group["is_after"]][0]
        print(f"Seed {seed}: {start_group['files'][0]} -> {end_group['files']}")
        result[seed] = {
            "seed": seed,
            "start_group_folder": start_group["path"],
            "end_group_folder": end_group["path"],
            "start_file": start_group["files"][0],
            "end_files": end_group["files"],
        }

    return result


# 1000448 까지. + 999936
# 2000384 까지. + 999,936
# 3000320 까지. 5860
# 1953 * 512 = 999,936

# 0, 1, ... 1953: 1954개
# 1954, ..., 3907: 1954개
# 3908, ..., 5861: 1954개
# 0 ~ 1,000,448 : 999,936

# 1,000,448 ~ 2,000,896; 999,936
# 2,000,896 ~ 2999808; 999,424
global_step_offsets = {
    9000448: 999936,
    8000000: 2000384,
    7000064: 3000320,
}
# max(global_step)
# 9000448
# 8000000
# 7000064


def concat_logs(file_A: str, file_B: str) -> Tuple[pd.DataFrame, int]:
    # Read CSV files into DataFrames
    df_A = pd.read_csv(file_A, index_col=0)
    df_B = pd.read_csv(file_B, index_col=0)

    # Calculate offsets
    max_global_step_of_df_B = df_B["global_step"].max()
    global_step_end = global_step_offsets[int(max_global_step_of_df_B)]
    global_step_offset = global_step_end + 512
    print(f"Global step offset: {global_step_offset}")

    # Cut the dataframe A.
    df_A = df_A[df_A["global_step"] <= global_step_end]

    # Calculate episode offset
    episode_offset = df_A["episode"].max()
    print(f"Episode offset: {episode_offset}")

    # Adjust B's values
    df_B["episode"] = df_B["episode"] + episode_offset
    df_B["global_step"] = df_B["global_step"] + global_step_offset

    # Update success counts and rates
    for i in range(4):
        success_count_col = f"{i}/success_count"
        visited_count_col = f"{i}/visited_count"
        success_rate_col = f"{i}/success_rate"

        df_A[visited_count_col] = df_A[success_count_col] / df_A[success_rate_col]
        df_B[visited_count_col] = df_B[success_count_col] / df_B[success_rate_col]

        max_visited_count = df_A[visited_count_col].max()
        max_success_count = df_A[success_count_col].max()

        df_B[visited_count_col] = df_B[visited_count_col] + max_visited_count
        df_B[success_count_col] = df_B[success_count_col] + max_success_count
        df_B[success_rate_col] = df_B[success_count_col] / df_B[visited_count_col]

    # Concatenate A and B
    df_combined = pd.concat([df_A, df_B], ignore_index=True).reset_index(drop=True)

    return df_combined, global_step_end


if __name__ == "__main__":
    # iterate_and_print_parsed()
    seed_groups = iterate_and_group_by_seed()
    print(seed_groups)
    output_directory = f"{current_canonical_directory}/../merged_entropy_241211/"
    os.makedirs(output_directory, exist_ok=True)
    for seed, info in seed_groups.items():
        start_file = info["start_file"]
        end_files = info["end_files"]
        start_group_folder = info["start_group_folder"]
        end_group_folder = info["end_group_folder"]
        for end_file in end_files:
            print(f"Seed {seed}: {start_file} -> {end_file}")
            path1 = os.path.abspath(
                os.path.join(base_dir, start_group_folder, start_file)
            )
            path2 = os.path.abspath(os.path.join(base_dir, end_group_folder, end_file))
            merged_log, transition_timestep = concat_logs(path1, path2)
            # save the file as csv
            output_path = f"{output_directory}/{seed}_{transition_timestep}.csv"
            merged_log.to_csv(output_path, index=True)
            print(f"Saved merged log to {output_path}")
