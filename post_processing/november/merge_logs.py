import os
import re
from collections import defaultdict
from typing import Optional, Dict

import pandas as pd

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)
base_dir = f"{current_canonical_directory}/../all_run_data241214-s2d-with-reward/"


def extract_info(string) -> Optional[Dict]:
    # Regular expression to match the desired components
    match = re.match(
        r"v33-room-v1-(sparse|dense)-False-seed(\d+)-from_(sparse|dense|)$", string
    )
    if match:
        return {
            "type": match.group(1),
            "seed": int(match.group(2)),
            "from": match.group(3) if match.group(3) else None,
            "path": string,
        }
    return None


def iterate_and_group_by_seed():
    seed_groups = defaultdict(list)
    # Iterate through directories and parse information
    for dir in os.listdir(base_dir):
        parsed = extract_info(dir)
        if parsed:
            seed_groups[parsed["seed"]].append(parsed)

    return seed_groups


def get_pairs_to_merge(seed_groups):
    pairs = []
    for seed, nodes in seed_groups.items():
        # Separate nodes into roots (from is None) and others
        roots = [node for node in nodes if node["from"] is None]
        children = [node for node in nodes if node["from"] is not None]

        # Find pairs to merge starting from roots
        for root in roots:
            find_pairs(root, children, pairs)
    return pairs


def find_pairs(current_node, children, pairs):
    # Find children of the current node
    for child in children:
        if child["from"] == current_node["type"]:
            pairs.append((current_node, child))


def iterate_and_print_parsed():
    base_dir = "../all_run_data241207/"
    for dir in os.listdir(base_dir):
        full_path = os.path.join(base_dir, dir)
        parsed = extract_info(dir)
        if parsed:
            print(
                f"{full_path} -> {parsed['from']} to {parsed['type']}; {parsed['seed']}"
            )
        else:
            print(f"{full_path} -> None")


def concat_logs(file_A: str, file_B: str):
    # Read CSV files into DataFrames
    df_A = pd.read_csv(file_A, index_col=0)
    df_B = pd.read_csv(file_B, index_col=0)

    # Calculate offsets
    episode_offset = df_A["episode"].max()
    global_step_offset = 3000320 + 512

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

    return df_combined


if __name__ == "__main__":
    # iterate_and_print_parsed()
    seed_groups = iterate_and_group_by_seed()
    pairs_to_merge = get_pairs_to_merge(seed_groups)
    output_directory = (
        f"{current_canonical_directory}/../merged_all_241214-s2d-with-reward"
    )
    for pair in pairs_to_merge:
        print(
            f"Merge {pair[0]['type']} (seed {pair[0]['seed']}) to {pair[1]['type']} (seed {pair[1]['seed']})"
        )
        print(pair[0]["path"] + " + " + pair[1]["path"])

        path1 = os.path.join(base_dir, pair[0]["path"])
        path2 = os.path.join(base_dir, pair[1]["path"])
        file_A = os.path.join(
            path1, [f for f in os.listdir(path1) if f.endswith(".csv")][0]
        )
        file_B = os.path.join(
            path2, [f for f in os.listdir(path2) if f.endswith(".csv")][0]
        )

        merged_log: pd.DataFrame = concat_logs(file_A, file_B)
        # save the file as csv
        os.makedirs(output_directory, exist_ok=True)
        output_path = f"{output_directory}/{pair[0]['type']}_to_{pair[1]['type']}_seed_{pair[1]['seed']}.csv"
        merged_log.to_csv(output_path, index=True)
        print(f"Saved merged log to {output_path}")
