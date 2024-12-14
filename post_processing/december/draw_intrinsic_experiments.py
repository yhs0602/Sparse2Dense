from collections import defaultdict
import os
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from post_processing.december.draw_figures_entropy import (
    calculate_group_average,
    calculate_group_std,
    running_average,
)
import scipy.stats as stats


def prepare_intrinsic_params(
    all_run_data_dir="all_run_data241211-icm",
) -> Tuple[Dict[str, List[Tuple[str, str]]], Dict[str, List[pd.DataFrame]], int]:
    room_groups = defaultdict(list)
    for group_name in os.listdir(all_run_data_dir):
        for file_name in os.listdir(os.path.join(all_run_data_dir, group_name)):
            # check if csv
            if not file_name.endswith(".csv"):
                continue
            file_path = os.path.join(all_run_data_dir, group_name, file_name)
            # find ir: the floating point number after "ir"
            ir = float(group_name.split("ir")[1])
            df = pd.read_csv(file_path)
            room_groups[str(ir)].append(df)
    axises = {
        "room": [
            ("global_step", "episode"),
            ("episode", "episode/length"),
            ("global_step", "0/success_rate"),
            ("global_step", "1/success_rate"),
            ("global_step", "2/success_rate"),
            ("global_step", "3/success_rate"),
            ("episode", "scaled_mean_intrinsic_rewards"),
            ("episode", "episode/reward"),
        ],
    }
    window_size = 80
    return axises, room_groups, window_size


def plot_groups_intrinsics(axis_x_name, axis_y_name, groups, groups_name, window_size):
    for group_name, group_data in groups.items():
        print(group_name)
        group_data: List[pd.DataFrame]
        # ffill the x_axis
        for i in range(len(group_data)):
            # data[axis_y_name] = data[axis_y_name].ffill()
            # data[axis_x_name] = data[axis_x_name].ffill()
            group_data[i].ffill(inplace=True)
            # group_data[i].loc[:, [axis_x_name, axis_y_name]] = group_data[i].loc[
            #     :, [axis_x_name, axis_y_name]
            # ].ffill()
            # Group for eval_episode and select smallest value
            if axis_y_name == "eval_episode":
                group_data[i][axis_y_name] = (
                    group_data[i]
                    .groupby(axis_x_name, as_index=False)
                    .agg({axis_y_name: "max"})[axis_y_name]
                )
            group_data[i].ffill(inplace=True)
            group_data[i].bfill(inplace=True)
            assert group_data[i][axis_y_name].isnull().sum() == 0
            assert group_data[i][axis_x_name].isnull().sum() == 0
            assert len(group_data[i][axis_x_name]) == len(group_data[i][axis_y_name])
            print(i)

        avg_data = calculate_group_average(group_data, axis_x_name, axis_y_name)
        std_data = calculate_group_std(group_data, axis_x_name, axis_y_name)

        window_size_to_use = window_size
        if axis_y_name != "scaled_mean_intrinsic_rewards":
            avg_data = running_average(avg_data, window_size_to_use)
            std_data = running_average(std_data, window_size_to_use)

        # Determine color based on the group
        if "0.1" in group_name:
            color = "#FF3935"  # 밝은 빨강
            label = "Intrinsic Motivation (0.1) + S2D"
        elif "0.01" in group_name:
            color = "#FFBB6A"  # 밝은 초록
            label = "Intrinsic Motivation (0.01) + S2D"
        else:
            raise ValueError(f"Unknown group: {group_name}")

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
            label=label,
            linewidth=5.0,
            color=color,
        )
        print("ploted")
        plt.fill_between(
            avg_data[axis_x_name],
            avg_data[axis_y_name] - std_data[axis_y_name],
            avg_data[axis_y_name] + std_data[axis_y_name],
            alpha=0.3,
            color=color,
        )
        print("filled")

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
                f"{groups_name}/{group_name};{axis_y_name}:        {slope:.2f}\\stdv{{{slope_std:.2f}}}"
            )
