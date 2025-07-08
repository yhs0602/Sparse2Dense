import os
import math
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import argparse

from post_processing.december.draw_normal_experiments import (
    plot_impl_normal_experiments,
    prepare_normal_params,
)


# Running Averaging Calculation Functions
def running_average(data, window_size: int):
    return data.rolling(window=window_size, min_periods=1).mean()


# Standard deviation calculation functions
def running_std(data, window_size):
    return data.rolling(window=window_size, min_periods=1).std()


# Calculate average data by group
def calculate_group_average(
    group_data: List[pd.DataFrame], x_axis: str, y_axis: str
) -> pd.DataFrame:
    concatenated = pd.concat(group_data)
    grouped = concatenated.groupby(x_axis).mean().reset_index()
    return grouped


# Calculate standard error by group
def calculate_group_std(
    group_data: List[pd.DataFrame], x_axis: str, y_axis: str
) -> pd.DataFrame:
    num_samples = len(group_data)
    concatenated = pd.concat(group_data)
    grouped = concatenated.groupby(x_axis).std().reset_index() / math.sqrt(num_samples)
    return grouped


def draw_ir_figures(
    axis_x_name,
    axis_y_name,
    groups: Dict[Tuple[str, str], List[str]],
    groups_name,
    window_size,
    normal_room_groups: Dict[str, List[pd.DataFrame]],
):
    print(f"Intrinsic Reward {groups_name}/{axis_x_name}/{axis_y_name}")

    for (algo, timing), files in groups.items():
        group_data: List[pd.DataFrame] = []
        for file in files:
            df = pd.read_csv(file, index_col=0)
            # Preprocess the data, ffill, etc...
            # Ffill the x_axis
            df.ffill(inplace=True)
            df.bfill(inplace=True)
            # assert df[axis_x_name].isnull().sum() == 0
            # assert df[axis_y_name].isnull().sum() == 0
            # assert len(df[axis_x_name]) == len(df[axis_y_name])

            group_data.append(df)

        # Data preparation done.
        # Now calculate the average and std of the group
        avg_data = calculate_group_average(group_data, axis_x_name, axis_y_name)
        std_data = calculate_group_std(group_data, axis_x_name, axis_y_name)

        avg_data = running_average(avg_data, window_size)
        std_data = running_average(std_data, window_size)

        # Determine color based on the group
        if "2M" in timing:
            color = "#E53935"  # 밝은 빨강
        elif "3M" in timing:
            color = "#66BB6A"  # 밝은 초록
        elif "1M" in timing:
            color = "#42A5F5"  # 밝은 파랑
        else:
            raise ValueError(f"Unknown group: {timing}")

        plt.rcParams.update(
            {
                "font.size": 30,
                "font.family": "Times new roman",
            }
        )
        plt.rcParams.update({"axes.linewidth": 3})
        plt.rcParams["font.weight"] = "bold"

        plt.plot(
            avg_data[axis_x_name],
            avg_data[axis_y_name],
            label=f"{algo}-{timing}",
            linewidth=5.0,
            # color=color,
        )
        # print("ploted")
        plt.fill_between(
            avg_data[axis_x_name],
            avg_data[axis_y_name] - std_data[axis_y_name],
            avg_data[axis_y_name] + std_data[axis_y_name],
            alpha=0.3,
            # color=color,
        )
        # print("filled")

        if "global_step" in axis_x_name and "episode" in axis_y_name:
            # print(f"{avg_data[axis_x_name]=}")
            # print(f"{avg_data[axis_y_name]=}")
            avg_data[axis_y_name] *= 100000
            # Report the slope and its std using the last point
            slope, intercept = np.polyfit(
                avg_data[axis_x_name], avg_data[axis_y_name], 1
            )
            # 2. 기울기의 표준 편차 계산
            regression_result = stats.linregress(
                avg_data[axis_x_name], avg_data[axis_y_name]
            )
            slope_std = regression_result.stderr
            # print(f"{groups_name}/{group_name};{axis_x_name}/{axis_y_name}: slope={slope:.2f}±{slope_std:.2f}")
            print(
                f"{algo}/{timing};{axis_y_name}:        {slope:.2f}\\stdv{{{slope_std:.2f}}}"
            )
    # Draw normal data
    for group, dfs in normal_room_groups.items():
        print(f"{group=}")
        for df in dfs:
            pass
    print("Done")


def main(is_pbim: bool):
    # For each folders in ir_runs
    # episode/episode_length
    # global_step/ 0..3/success_rate
    # global_step / episode

    # Group by: (algorithm, timing)

    grouped_data = {}

    this_dir = os.path.dirname(os.path.abspath(__file__))
    ir_runs_dir = f"{this_dir}/ir_runs_full_pbim" if is_pbim else f"{this_dir}/ir_runs"

    for algo in os.listdir(ir_runs_dir):
        if algo == ".DS_Store":
            continue
        merged_dir = f"{ir_runs_dir}/{algo}/merged"
        for file in os.listdir(merged_dir):
            seed = file.split("-")[0]
            timing = file.split("_")[-1].split(".")[0]
            if (algo, timing) not in grouped_data:
                grouped_data[(algo, timing)] = []
            grouped_data[(algo, timing)].append(f"{merged_dir}/{file}")

    # Sort grouped_data by algo
    grouped_data = dict(
        sorted(
            grouped_data.items(),
            key=lambda x: (x[0][0], x[0][1]),
        )
    )

    for (algo, timing), files in grouped_data.items():
        print(f"{algo=}, {timing=}, {len(files)=}")

    # Filter
    # e3b = 2M, icm = 1M ngu = 1M
    # Filter the grouped_data
    filtered_grouped_data: Dict[Tuple[str, str], List[str]] = {}
    for (algo, timing), files in grouped_data.items():
        if algo == "e3b":  #  and timing == "2M":
            filtered_grouped_data[(algo, timing)] = files
        elif algo == "icm":  # and timing == "1M":
            filtered_grouped_data[(algo, timing)] = files
        elif algo == "ngu":  # and timing == "1M":
            filtered_grouped_data[(algo, timing)] = files

    ###########
    # Prepare normal data
    normal_axises, normal_room_groups, normal_window_size = prepare_normal_params(
        f"{this_dir}/../post_processing/merged_all_241214-s2d-with-reward"
    )
    normal_room_groups: Dict[str, List[pd.DataFrame]]
    ###########

    # Draw graph by grouped data
    axis_list = [
        ("episode", "episode/length"),
        ("global_step", "0/success_rate"),
        ("global_step", "1/success_rate"),
        ("global_step", "2/success_rate"),
        ("global_step", "3/success_rate"),
        ("global_step", "episode"),
    ]

    for axis in axis_list:
        plt.figure(figsize=(14, 8))
        draw_ir_figures(
            axis[0],
            axis[1],
            filtered_grouped_data,
            algo,
            window_size=normal_window_size,
            normal_room_groups=normal_room_groups,
        )
        try:
            plot_impl_normal_experiments(
                axis[0],
                axis[1],
                normal_room_groups,
                "normal",
                normal_window_size,
            )
        except KeyError as e:
            print(f"KeyError | AssertionError: {axis[0]}, {axis[1]}, {e}")
        plt.title(f"{axis[0]}/{axis[1]}")
        plt.xlabel(axis[0])
        plt.ylabel(axis[1])
        plt.legend()
        # plt.show()
        filename = f"{ir_runs_dir}/../ir_full_pbim_figs_filtered_all/{axis[0].replace('/', '_')}_{axis[1].replace('/', '_')}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbim", action="store_true")
    args = parser.parse_args()
    main(args.pbim)
