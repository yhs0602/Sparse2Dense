import json
import math
import os
from typing import List

from PIL import Image
import numpy as np
import pandas as pd
from download_s2d_trajectories import get_run


def render_group(file_paths: List[str], radius, transition_timing, timing):
    image = Image.new("RGBA", (700, 1000), color=(255, 255, 255, 255))
    pixels = image.load()
    aggregations = {}

    start_positions = []
    end_positions = []

    for full_file_path in file_paths:
        df = pd.read_csv(full_file_path)
        # For each row, get the episode/positions and json.loads
        for _, row in df.iterrows():
            positions = json.loads(row["episode/positions"])
            if not positions:
                continue
            start_positions.append(positions[0])
            end_positions.append(positions[-1])
            for position in positions:
                # print(position)
                x, y, z, yaw = position
                x = int(x * 50)
                z = int(z * 50)
                aggregations[(x, z)] = aggregations.get((x, z), 0) + 1
    max_count = max(aggregations.values())
    for (x, z), count in aggregations.items():
        # alpha = int(min(count / max_count * 1000, 255))
        scaled = math.log1p(count) / math.log1p(max_count)
        scaled = scaled**0.5
        alpha = int(scaled * 255)
        pixels[x, z] = (0, 0, 0, alpha)

    for start_position in start_positions:
        x, y, z, yaw = start_position
        x = int(x * 50)
        z = int(z * 50)
        # draw a rectangle with radius
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dz * dz <= radius * radius:
                    nx, nz = x + dx, z + dz
                    if nx < 0 or nx >= 700 or nz < 0 or nz >= 1000:
                        continue
                    pixels[nx, nz] = (255, 0, 0, 255)

    for end_position in end_positions:
        x, y, z, yaw = end_position
        x = int(x * 50)
        z = int(z * 50)
        # draw a rectangle with radius
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if dx * dx + dz * dz <= radius * radius:
                    nx, nz = x + dx, z + dz
                    if nx < 0 or nx >= 700 or nz < 0 or nz >= 1000:
                        continue
                    pixels[nx, nz] = (0, 0, 255, 255)

    image.save(f"trajectory_images-s2d/s2d_{transition_timing}_timing_{timing}.png")


# 목표
# 1. 시드 6종에 대해
# 2. 트랜지션 타이밍 100만, 200만, 300만, 400만, 500만에 대해
# 3. 각각 군에 대한 0만 100만 200만 300만 ... 1000만 스텝에서의 포지션들을 렌더링하기.
# 결과물은 (5개의 실험군) * (11개의 단계) 총 55개의 그림.
def sample_trajectories():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, "all_run_trajectories-s2d")
    run_id_to_file_path = {}
    for group_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, group_dir)):
            continue
        group_path = os.path.join(data_dir, group_dir)
        for file in os.listdir(group_path):
            if not file.endswith(".csv.gz"):
                continue
            file_path = os.path.join(group_path, file)
            run_id = file.split(".")[0]
            run_id_to_file_path[run_id] = file_path

    for run_id, file_path in run_id_to_file_path.items():
        print(f"{run_id} -> {file_path}")

    sampled_dir = os.path.join(this_dir, "sampled_trajectories-s2d")
    os.makedirs(sampled_dir, exist_ok=True)
    # sample trajectories to files
    for run_id, file_path in run_id_to_file_path.items():
        df = pd.read_csv(file_path)
        global_steps = df["global_step"].values
        max_step = max(global_steps)
        for target in range(0, 10000000, 1000000):
            if target - 50000 > max_step:
                print(f"{run_id} {target} > {max_step} is too large")
                break
            result_rows = []
            idx = np.searchsorted(global_steps, target)
            start = max(0, idx - 15)
            end = min(len(global_steps), idx + 15)
            result_rows.append(df.iloc[start:end])
            result_df = pd.concat(result_rows, ignore_index=True)
            result_df.to_csv(
                os.path.join(sampled_dir, f"{run_id}_{target}.csv.gz"),
                index=False,
                compression="gzip",
            )
            print(f"Saved {len(result_df)} trajectories to {target}")
    return


def render_trajectories():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, "sampled_trajectories-s2d")
    run_id_to_file_path = {}
    for file in os.listdir(data_dir):
        if not file.endswith(".csv.gz"):
            continue
        file_path = os.path.join(data_dir, file)
        run_id_and_timing = file.split(".")[0]
        run_id, timing = run_id_and_timing.split("_")
        timing = int(timing)
        run_id = run_id.strip()
        run_id_to_file_path[(run_id, timing)] = file_path

    for (run_id, timing), file_path in run_id_to_file_path.items():
        print(f"{run_id} {timing} -> {file_path}")

    output_dir = os.path.join(this_dir, "trajectory_images-s2d")
    os.makedirs(output_dir, exist_ok=True)

    for transition_timing in [1, 2, 3, 4, 5]:
        for timing in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            file_paths = []
            for seed in [0, 42, 9876, 7777, 2024, 1234]:
                run_id = get_run(seed, transition_timing, timing)

                # (100만트랜지션에서 0스텝 -> s에 0, 100만스텝 -> s에 100만, 200만스텝 -> d에 100만)
                # Adjust the offsets to extract.
                if timing <= transition_timing:
                    # use plain timing offset. Consider 3M 4M 5M
                    if timing >= 3:
                        offset = (timing - 3) * 1000000
                    else:
                        offset = timing * 1000000
                else:
                    offset = (timing - transition_timing) * 1000000
                run_file = run_id_to_file_path[(run_id, offset)]
                print(
                    f"Loading {run_file.split('/')[-1]} for {seed=} {transition_timing=} {timing=} {offset=}"
                )
                file_paths.append(run_file)
                # df = pd.read_csv(run_file)
            render_group(
                file_paths, radius=3, transition_timing=transition_timing, timing=timing
            )

            # print(f"{timing}/{transition_timing}: {run_ids=}")

            # file_paths = [run_id_to_file_path[run_id] for run_id in run_ids]
            # print(f"{timing}/{transition_timing}: {file_paths=}")
            # render_group(file_paths, radius=3)


if __name__ == "__main__":
    render_trajectories()
