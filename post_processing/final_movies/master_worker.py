import json
import os
from typing import List, Dict

import wandb
from tqdm import tqdm
from wandb.apis.public import Run

from post_processing.alpha_trajectory import create_trajectory_image
from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

current_file_dir = os.path.dirname(os.path.abspath(__file__))
room_out_dir = os.path.join(current_file_dir, "room_media")
os.makedirs(room_out_dir, exist_ok=True)


def get_run(run: Run, keys: List[str], episode: id) -> List[Dict]:
    data = run.scan_history(keys=keys)
    data = [row for row in data if int(row["episode"]) == episode]
    # data.to_csv(run_csv_path, compression="gzip")
    return data


def main():
    room_runs = {
        "sparse": ["8avm2wm9", 1025],
        "d2s": ["24gpf0l2", 1457],
        "dense1": ["17xr5n4c", 1001],
        "dense2": ["le1x2bzj", 1662],
        "s2d1": ["7m1bq6ml", 2569],
        "s2d2": ["x6xtvwv5", 2729],
    }
    cross_runs = {
        "d2s-0-2": ["btlv4am1", 6219],
        "d2s-1-0": ["cb3fhbdg", 5482],
        "d2s-2-1": ["yg310psi", 4945],
        "dense-0-1": ["xy8clqb9", 5601],  # forward
        # "dense-0-2": ["7557y1ep", 4882],  # right
        "dense-1-2": ["i03plgdq", 6384],
        "dense-2-0": ["df334mix", 4128],  #
        "sparse-0-0": ["ngk59t0e", 6024],  # left
        # "sparse-0-1": ["rn06swvu", 6146], # forward
        "sparse-1-2": ["1ezz5e78", 5184],  # right
        "sparse-2-1": ["3r4ctsay", 5700],  # forward
        "s2d-0-1": ["p6fa3bvp", 5627],  # 2M, front
        "s2d-1-2": ["edsz2myn", 8563],  # 2M, right
        "s2d-2-0": ["yxbvuc6o", 7038],  # 2M, left
    }

    api = wandb.Api(timeout=180)
    total_runs = {}
    total_runs.update(room_runs)
    total_runs.update(cross_runs)
    for reward, run_tuple in tqdm(total_runs.items()):
        run_id = run_tuple[0]
        episode = run_tuple[1]
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
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
        data: List[Dict] = get_run(run, keys, episode)
        row = data[0]
        episode_id = row["episode"]
        print(f"Found episode: {episode_id}")
        # 1. create trajectory images
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
        image_out_filename = os.path.join(
            room_out_dir, f"{run.group}_{run.id}_{episode_id}.png"
        )
        create_trajectory_image(
            positions,
            image_out_filename,
            goals=goal1,
            min_x=min_x,
            min_z=min_z,
            max_x=max_x,
            max_z=max_z,
        )
        # 2. Reverse action from trajectories
        print("Reversing action...")
        if not isinstance(positions, list):
            positions = json.loads(positions)
        # Get action from positions.
        # Action = TURN_LEFT_90, TURN_RIGHT_90, MOVE_FORWARD
        # positions: [(x, y, z, yaw), ...]
        # Use yaw and position to get action
        actions = []
        for i in range(1, len(positions)):
            x, y, z, yaw = positions[i]
            prev_x, prev_y, prev_z, prev_yaw = positions[i - 1]
            if yaw == prev_yaw:
                actions.append(0)  # MOVE_FORWARD
            elif yaw == prev_yaw + 90:
                actions.append(2)  # TURN_RIGHT_90
            elif yaw == prev_yaw - 90:
                actions.append(1)  # TURN_LEFT_90
            else:
                print(f"Unknown action: {yaw} {prev_yaw}")
        # 3. Create egocentric movies
        selected_trajectory = {
            "positions": positions,
            "actions": actions,
            "goal": goal1,
        }
        inverse_out_filename = os.path.join(
            room_out_dir, f"{run.group}_{run.id}_{episode_id}.json"
        )
        with open(inverse_out_filename, "w") as f:
            json.dump(selected_trajectory, f)


if __name__ == "__main__":
    main()
