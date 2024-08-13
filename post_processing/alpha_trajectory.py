# 13 x 13
import colorsys
import json
import os
from collections import deque
from typing import Tuple, List

import pandas as pd
import wandb
from PIL import ImageDraw, Image, ImageChops
from tqdm import tqdm
from wandb.apis.public import Run

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT


# 0, 0 -> 12, 19
# 6.0, ~, 9.5


def mc_coord_to_image_coord(x, z, max_x, max_z) -> Tuple[float, float]:
    return z, max_x - x


def hsl_to_rgb(h, s, l):
    return tuple(round(i * 255) for i in colorsys.hls_to_rgb(h, l, s))


def get_color(
    step, total_steps, base_color: Tuple[int, int, int], alpha: int
) -> Tuple[int, int, int, int]:
    # base_color는 (R, G, B) 형식
    h, l, s = colorsys.rgb_to_hls(
        base_color[0] / 255, base_color[1] / 255, base_color[2] / 255
    )
    lightness = min(step / total_steps, 0.7)
    r, g, b = hsl_to_rgb(h, s, lightness)
    return r, g, b, alpha


def create_trajectory_image(positions, filename, goals, min_x, max_x, min_z, max_z):
    cell_size = 20
    agent_size_in_cell = 0.3
    goal_size_in_cell = 0.1
    # get grid size
    # for Mc: traverse from (min_x, min_z) Left bottom to (max_x, max_z) Right top
    # Image: (min_z, max_x - min_x) ~ (max_z, 0) to navigate around
    grid_x_length = max_z - min_z
    grid_z_length = max_x - min_x

    print(f"{grid_x_length} x {grid_z_length}")

    # PIL로 Image 생성
    img = Image.new(
        "RGBA",
        (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
        color="white",
    )
    img.putalpha(0)
    # Grid Draw
    draw = ImageDraw.Draw(img, "RGBA")

    # for x in range(grid_x_length + 1):
    #     draw.line(
    #         [(x * cell_size, 0), (x * cell_size, grid_z_length * cell_size)],
    #         fill="black",
    #         width=1,
    #     )
    # for z in range(grid_z_length + 1):
    #     draw.line(
    #         [(0, z * cell_size), (grid_x_length * cell_size, z * cell_size)],
    #         fill="black",
    #         width=1,
    #     )
    # Goal Draw
    # Goal squeeze
    goals = [(goal[0], goal[2]) if len(goal) == 3 else goal for goal in goals]

    image_goal_coords = [
        mc_coord_to_image_coord(goal[0], goal[1], max_x, max_z) for goal in goals
    ]
    image_positions = [
        mc_coord_to_image_coord(pos[0], pos[2], max_x, max_z) for pos in positions
    ]
    print(f"goal: {goals} -> {image_goal_coords}")
    for goal in image_goal_coords:
        draw.ellipse(
            [
                (goal[0] - goal_size_in_cell) * cell_size,
                (goal[1] - goal_size_in_cell) * cell_size,
                (goal[0] + goal_size_in_cell) * cell_size,
                (goal[1] + goal_size_in_cell) * cell_size,
            ],
            fill="green",
        )
    # Trajectory Draw
    # for pos in positions:
    #     tmp_img = Image.new(
    #         "RGBA",
    #         (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
    #     )
    #     tmp_img.putalpha(0)
    #     tmp_draw = ImageDraw.Draw(tmp_img, "RGBA")
    #     tmp_draw.ellipse(
    #         [
    #             (pos[0] - grid_min_x - agent_size_in_cell) * cell_size,
    #             (pos[1] - grid_min_z - agent_size_in_cell) * cell_size,
    #             (pos[0] - grid_min_x + agent_size_in_cell) * cell_size,
    #             (pos[1] - grid_min_z + agent_size_in_cell) * cell_size,
    #         ],
    #         fill=(117, 0, 0, 10),
    #     )
    #     img = Image.alpha_composite(img, tmp_img)
    #     del tmp_img, tmp_draw
    # 알파값은 최근일수록 255, 이전일수록 0
    last_index = len(positions) - 1
    tmp_imgs = deque(maxlen=3)
    for i in range(len(positions) - 1):
        # alpha = max(int(255 * i / last_index), 20)
        # k = 3
        # alpha = min(
        #     max(int(255 * (math.exp(k * i / last_index) - 1) / (math.exp(1) - 1)), 20),
        #     255,
        # )
        tmp_img = Image.new(
            "RGBA",
            (grid_x_length * cell_size + 1, grid_z_length * cell_size + 1),
        )
        tmp_img.putalpha(0)
        tmp_draw = ImageDraw.Draw(tmp_img, "RGBA")
        tmp_draw.line(
            [
                (image_positions[i][0]) * cell_size,
                (image_positions[i][1]) * cell_size,
                (image_positions[i + 1][0]) * cell_size,
                (image_positions[i + 1][1]) * cell_size,
            ],
            fill=get_color(i, last_index, (0, 255, 0), 120),  # (0, 255, 255, alpha),
            width=int(cell_size * agent_size_in_cell),
            joint="curve",
        )
        tmp_imgs.append(tmp_img)
        # AB + BC - (AB ^ BC) 해야 함.
        if len(tmp_imgs) == 2:
            newer_img = tmp_imgs[1]
            older_img = tmp_imgs[0]
            tmp_img = ImageChops.subtract(newer_img, older_img)
        elif len(tmp_imgs) == 3:
            tmp_img = ImageChops.subtract(
                tmp_imgs[2], ImageChops.add(tmp_imgs[0], tmp_imgs[1])
            )
        else:
            tmp_img = tmp_imgs[0]
        img = Image.alpha_composite(img, tmp_img)
        del tmp_img, tmp_draw
    # 끝점 Draw
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse(
        [
            (image_positions[-1][0] - agent_size_in_cell) * cell_size,
            (image_positions[-1][1] - agent_size_in_cell) * cell_size,
            (image_positions[-1][0] + agent_size_in_cell) * cell_size,
            (image_positions[-1][1] + agent_size_in_cell) * cell_size,
        ],
        fill="red",
    )
    # 시작점 Draw
    draw.ellipse(
        [
            (image_positions[0][0] - agent_size_in_cell) * cell_size,
            (image_positions[0][1] - agent_size_in_cell) * cell_size,
            (image_positions[0][0] + agent_size_in_cell) * cell_size,
            (image_positions[0][1] + agent_size_in_cell) * cell_size,
        ],
        fill="blue",
    )
    img.save(filename)


def get_run(run: Run, keys: List[str]) -> pd.DataFrame:
    cache_dir = "cache"
    run_csv_path = os.path.join(cache_dir, f"{run.id}.csv.gz")
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(run_csv_path):
        data = run.history(keys=keys)
        data.to_csv(run_csv_path, compression="gzip")
        return data
    else:
        return pd.read_csv(run_csv_path, compression="gzip")


def make_room_trajectory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "trajectory_images")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B API
    api = wandb.Api(timeout=180)
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    groups = [
        "v30-crossw2-transition-2000000-0",
        "v30-crossw2-d2s-3000000-0",
        "v30-crossw2-dense-0",
        "v30-crossw2-sparse-0",
        "v30-crossw2-transition-2000000-1",
        "v30-crossw2-d2s-2000000-1",
        "v30-crossw2-dense-1",
        "v30-crossw2-sparse-1",
        "v30-crossw2-transition-2000000-2",
        "v30-crossw2-d2s-2000000-2",
        "v30-crossw2-dense-2",
        "v30-crossw2-sparse-2",
        "v31-room-v1-transition-3000000",
        "v31-room-v1-d2s-3000000",
        "v31-room-v1-sparse",
        "v31-room-v1-dense",
    ]
    group_runs = [run for run in runs if run.group in groups]
    print(f"Found {len(group_runs)} runs")

    for run in tqdm(group_runs):
        # select keys, min_max based on the group
        if "crossw2" in run.group:
            keys = [
                "episode/positions",
                "episode",
                "episode/goal",
            ]
            min_x = 1
            max_x = 14
            min_z = 0
            max_z = 13
        elif "room" in run.group:
            keys = [
                "episode/positions",
                "episode/spawn",
                "episode",
                "episode/goal",
            ]
            min_x = 0
            max_x = 12
            min_z = 0
            max_z = 19
        else:
            print(f"Unknown group: {run.group}")
            continue
        # Get the log data
        data: pd.DataFrame = get_run(run, keys)
        # Generate videos for each episode
        n = 0
        for i in range(len(data) - 1, -1, -1):
            row = data.iloc[i]
            episode_id = int(row["episode"])
            positions = row["episode/positions"]
            if not isinstance(positions, list):
                positions = json.loads(positions)
            goal1 = row["episode/goal"]
            if not isinstance(goal1, (list, tuple)):
                goal1 = json.loads(goal1)
            print(f"Goal:{goal1}")
            print(f"Start:{positions[0]}")
            if not isinstance(goal1[0], list):
                goal1 = [goal1]
            out_filename = os.path.join(
                output_dir, f"{run.group}_{run.id}_{episode_id}.png"
            )
            create_trajectory_image(
                positions,
                out_filename,
                goals=goal1,
                min_x=min_x,
                min_z=min_z,
                max_x=max_x,
                max_z=max_z,
            )
            n += 1
            if n >= 3:
                break
        else:
            print(f"No data for run {run.id} in {run.group}")


if __name__ == "__main__":
    make_room_trajectory()
