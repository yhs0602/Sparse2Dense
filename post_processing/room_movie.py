# 13 x 13

import wandb

from post_processing.movie import create_video_from_positions
from room.room_env import real_room_str, room_palette
from wandb_envs import WANDB_PROJECT, WANDB_ENTITY


def make_room_movie():
    # Initialize W&B API
    api = wandb.Api(timeout=120)

    # Select the project and run
    run_names = [
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/w8r7kqnu",  # transition 2M
    ]
    for run_name in run_names:
        run = api.run(run_name)
        # Get the log data
        data = run.history(
            keys=[
                "episode/positions",
                "episode/spawn",
                "episode",
                # "eval_reached_goal",
                "episode/goal",
            ],
            pandas=False,
        )
        # Generate videos for each episode
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            reached_goal = True  # Unkown
            goal1 = episode_data["episode/goal"]
            print(f"Goal:{goal1}")
            create_video_from_positions(
                real_room_str,
                1,
                0,
                positions,
                f"3movoo_{run.id}_{episode_id}.mp4",
                episode=episode_data["episode"],
                goals=[(int(goal1[0]), int(goal1[2]))],
                reached_goal=reached_goal,
                palette=room_palette,
            )
            n += 1
            if n >= 3:
                break
        else:
            print("No data")


if __name__ == "__main__":
    make_room_movie()
