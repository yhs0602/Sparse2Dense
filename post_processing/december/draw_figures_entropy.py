import math
import os
from collections import defaultdict
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# Running Averaging Calculation Functions
def running_average(data, window_size: int):
    return data.rolling(window=window_size, min_periods=1).mean()


# Standard deviation calculation functions
def running_std(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()


# Loading data by group
def load_group_data(group_dir) -> List[pd.DataFrame]:
    group_data = []
    for root, _, files in os.walk(group_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                group_data.append(data)
    return group_data


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


# Visualisation functions, std err
def plot_groups(
    groups: Dict[str, List[pd.DataFrame]],
    groups_name: str,
    axis: Tuple[str, str],
    window_size=10,
):
    axis_x_name = axis[0]
    axis_y_name = axis[1]

    plt.figure(figsize=(14, 8))

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

        avg_data = running_average(avg_data, window_size)
        std_data = running_average(std_data, window_size)

        # Determine color based on the group
        if "sparse_dense" in group_name:
            color = "#AE4338"  # 174 67 56 red
        elif "sparse_sparse" in group_name:
            color = "#57A148"  # 87 161 72 green
        elif "dense_dense" in group_name:
            color = "#5D83D8"  # 93 131 216 blue
        elif "dense_sparse" in group_name:
            color = "#A68460"  # 166 132 96 brown
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
            label=f"group: {group_name}",
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

    # plt.xlabel(axis_x_name)
    # plt.ylabel(axis_y_name)
    # plt.legend()
    # plt.title(f"{axis_x_name} vs {axis_y_name}")
    y_max = 8000
    y_min = 0
    if "rate" in axis_y_name:
        y_max = 1
    elif "length" in axis_y_name:
        y_max = 20000
    elif "eval_episode" in axis_y_name:
        y_max = 1000
    if "success_rate" in axis_y_name:
        x_max = 10000000  # 5000000
        y_min = 0.4
    else:
        x_max = None
    plt.ylim(bottom=y_min)  # , top=y_max
    plt.xlim(left=0, right=x_max)
    plt.grid(True)

    # plt.show()

    axis_x_name = axis_x_name.replace("/", "_")
    axis_y_name = axis_y_name.replace("/", "_")
    figure_path = f"./figures/{groups_name}-{axis_x_name}_{axis_y_name}.png"
    plt.savefig(figure_path, dpi=300)
    print(f"Saved {figure_path}")


def main():
    all_run_data_dir = "../december/merged_241204"
    room_groups = defaultdict(list)
    for file_name in os.listdir(all_run_data_dir):
        # check if csv
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(all_run_data_dir, file_name)
        # 파일 이름에서 'from'과 'to' 추출
        parts = file_name.split("_")
        from_type = parts[0]  # dense 또는 sparse
        to_type = parts[2]  # dense 또는 sparse

        if from_type == "dense":
            if "7777" in file_name or "2024" in file_name:
                continue

        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(file_path)

        # (from, to) 키를 사용하여 딕셔너리에 DataFrame 추가
        room_groups[f"{from_type}_{to_type}"].append(df)
    # Plot
    # cross:
    # Grouping based on the end number of the group.
    # episode -> episode/length
    # global_step -> episode
    # eval_episode -> eval_episode/length
    # global_step -> eval_episode
    # room:
    axises = {
        "cross": [
            ("episode", "episode/length"),
            ("global_step", "episode"),
            ("eval_episode", "eval_episode/length"),
            ("global_step", "eval_episode"),
        ],
        "room": [
            ("global_step", "episode"),
            ("episode", "episode/length"),
            ("global_step", "0/success_rate"),
            ("global_step", "1/success_rate"),
            ("global_step", "2/success_rate"),
            ("global_step", "3/success_rate"),
        ],
    }
    # Group by eval_idx and room
    window_size = 80
    # for cross_axis in axises["cross"]:
    #     plot_groups(cross_0_groups, "cross_0", cross_axis, window_size=window_size)
    #     plot_groups(cross_1_groups, "cross_1", cross_axis, window_size=window_size)
    #     plot_groups(cross_2_groups, "cross_2", cross_axis, window_size=window_size)
    for room_axis in axises["room"]:
        plot_groups(room_groups, "room", room_axis, window_size=window_size)
        print(f"Plotted {room_axis}")


if __name__ == "__main__":
    main()
