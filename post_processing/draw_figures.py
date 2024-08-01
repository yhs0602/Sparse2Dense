import math
import os
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


# 러닝 애버리지 계산 함수
def running_average(data, window_size: int):
    return data.rolling(window=window_size, min_periods=1).mean()


# 표준 편차 계산 함수
def running_std(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()


# 그룹별 데이터 로드
def load_group_data(group_dir) -> List[pd.DataFrame]:
    group_data = []
    for root, _, files in os.walk(group_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                group_data.append(data)
    return group_data


# 그룹별 평균 데이터 계산
def calculate_group_average(
    group_data: List[pd.DataFrame], x_axis: str, y_axis: str
) -> pd.DataFrame:
    concatenated = pd.concat(group_data)
    grouped = concatenated.groupby(x_axis).mean().reset_index()
    return grouped


# 그룹별 standard error 계산
def calculate_group_std(
    group_data: List[pd.DataFrame], x_axis: str, y_axis: str
) -> pd.DataFrame:
    num_samples = len(group_data)
    concatenated = pd.concat(group_data)
    grouped = concatenated.groupby(x_axis).std().reset_index() / math.sqrt(num_samples)
    return grouped


# 시각화 함수, std err
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
        group_data: List[pd.DataFrame]
        # ffill the x_axis
        for i in range(len(group_data)):
            # data[axis_y_name] = data[axis_y_name].ffill()
            # data[axis_x_name] = data[axis_x_name].ffill()
            group_data[i].ffill(inplace=True)
            # group_data[i].loc[:, [axis_x_name, axis_y_name]] = group_data[i].loc[
            #     :, [axis_x_name, axis_y_name]
            # ].ffill()
            # eval_episode에 대해 그룹화하고 가장 작은 값 선택
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

        avg_data = calculate_group_average(group_data, axis_x_name, axis_y_name)
        std_data = calculate_group_std(group_data, axis_x_name, axis_y_name)

        avg_data = running_average(avg_data, window_size)
        std_data = running_average(std_data, window_size)

        # Determine color based on the group
        if "transition" in group_name:
            color = "#AE4338"  # 174 67 56 red
        elif "sparse" in group_name:
            color = "#57A148"  # 87 161 72 green
        elif "dense" in group_name:
            color = "#5D83D8"  # 93 131 216 blue
        elif "d2s" in group_name:
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
        plt.fill_between(
            avg_data[axis_x_name],
            avg_data[axis_y_name] - std_data[axis_y_name],
            avg_data[axis_y_name] + std_data[axis_y_name],
            alpha=0.3,
            color=color,
        )

        if "global_step" in axis_x_name and "episode" in axis_y_name:
            # print(f"{avg_data[axis_x_name]=}")
            # print(f"{avg_data[axis_y_name]=}")
            avg_data[axis_y_name] *= 100000
            # Report the slope and its std using the last point
            slope, intercept = np.polyfit(avg_data[axis_x_name], avg_data[axis_y_name], 1)
            # 2. 기울기의 표준 편차 계산
            regression_result = stats.linregress(avg_data[axis_x_name], avg_data[axis_y_name])
            slope_std = regression_result.stderr
            # print(f"{groups_name}/{group_name};{axis_x_name}/{axis_y_name}: slope={slope:.2f}±{slope_std:.2f}")
            print(f"{groups_name}/{group_name};{axis_y_name}:        {slope:.2f}\\stdv{{{slope_std:.2f}}}")

    # plt.xlabel(axis_x_name)
    # plt.ylabel(axis_y_name)
    # plt.legend()
    # plt.title(f"{axis_x_name} vs {axis_y_name}")
    y_max = 8000
    if "rate" in axis_y_name:
        y_max = 1
    elif "length" in axis_y_name:
        y_max = 20000
    elif "eval_episode" in axis_y_name:
        y_max = 1000
    plt.ylim(bottom=0)  # , top=y_max
    plt.xlim(left=0)
    plt.grid(True)

    # plt.show()

    axis_x_name = axis_x_name.replace("/", "_")
    axis_y_name = axis_y_name.replace("/", "_")
    figure_path = f"./figures/{groups_name}-{axis_x_name}_{axis_y_name}.png"
    plt.savefig(figure_path, dpi=450)


# 메인 함수
def main():
    all_run_data_dir = "./all_run_data"
    cross_0_groups = {}
    cross_1_groups = {}
    cross_2_groups = {}
    room_groups = {}

    useful_groupnames = {
        "cross_0-global_step_episode": (
            # "v30-crossw2-transition-2000000-0",
            "v30-crossw2-transition-1000000-0",
            "v30-crossw2-transition-3000000-0",
            # "v30-crossw2-d2s-3000000-0",
            "v30-crossw2-d2s-2000000-0",
            "v30-crossw2-d2s-1000000-0",
            # "v30-crossw2-dense-0",
            # "v30-crossw2-sparse-0",
        ),
        "cross_1-global_step_episode": (
            "v30-crossw2-transition-1000000-1",
            "v30-crossw2-transition-3000000-1",
            # "v30-crossw2-transition-2000000-1",
            # "v30-crossw2-d2s-2000000-1",
            "v30-crossw2-d2s-1000000-1",
            "v30-crossw2-d2s-3000000-1",
            # "v30-crossw2-dense-1",
            # "v30-crossw2-sparse-1",
        ),
        "cross_2-global_step_episode": (
            # "v30-crossw2-transition-2000000-2",
            "v30-crossw2-transition-1000000-2",
            "v30-crossw2-transition-3000000-2",
            # "v30-crossw2-d2s-2000000-2",
            "v30-crossw2-d2s-1000000-2",
            "v30-crossw2-d2s-3000000-2",
            # "v30-crossw2-dense-2",
            # "v30-crossw2-sparse-2",
        ),
        "room": (
            "v31-room-v1-transition-1000000",
            "v31-room-v1-transition-2000000",
            # "v31-room-v1-transition-3000000",
            # "v31-room-v1-d2s-3000000",
            "v31-room-v1-d2s-2000000",
            "v31-room-v1-d2s-1000000",
            # "v31-room-v1-sparse",
            # "v31-room-v1-dense",
        ),
    }
    flatten_groupnames = [
        groupname
        for groupnames in useful_groupnames.values()
        for groupname in groupnames
    ]

    for group_dir in os.listdir(all_run_data_dir):
        if group_dir not in flatten_groupnames:
            continue
        group_path = os.path.join(all_run_data_dir, group_dir)
        if os.path.isdir(group_path):
            group_data = load_group_data(group_path)
            if "crossw2" in group_dir:
                if group_dir.endswith("0"):
                    cross_0_groups[group_dir] = group_data
                elif group_dir.endswith("1"):
                    cross_1_groups[group_dir] = group_data
                elif group_dir.endswith("2"):
                    cross_2_groups[group_dir] = group_data
                else:
                    raise ValueError(f"Unknown group: {group_dir}")
            elif "room" in group_dir:
                room_groups[group_dir] = group_data
            else:
                raise ValueError(f"Unknown group: {group_dir}")
    # Plot
    # cross:
    # 그룹의 끝 번호를 기반으로 그룹핑.
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
    for cross_axis in axises["cross"]:
        plot_groups(cross_0_groups, "cross_0", cross_axis, window_size=window_size)
        plot_groups(cross_1_groups, "cross_1", cross_axis, window_size=window_size)
        plot_groups(cross_2_groups, "cross_2", cross_axis, window_size=window_size)
    for room_axis in axises["room"]:
        plot_groups(room_groups, "room", room_axis, window_size=window_size)


if __name__ == "__main__":
    main()
