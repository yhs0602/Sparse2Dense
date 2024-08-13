# 13 x 13
import wandb

from management.make_h_movie import create_video_from_positions
from wandb_envs import WANDB_PROJECT

cross_str = [
    "xxxxxoooxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "ooooooxoooooo",
    "oxxxxxxxxxxxo",
    "ooooooxoooooo",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoooxxxxx",
]

assert len(cross_str) == 13


# start point: 2 32 6 > 0 0 0


def make_cross_movie():
    # Initialize W&B API
    api = wandb.Api(timeout=30)

    # Select the project and run
    project_name = WANDB_PROJECT
    run_id = "af0auegt"  # 5lf40vyr: s2d
    run = api.run(f"{project_name}/{run_id}")

    # Get the log data
    data = run.history(keys=["episode/positions"], pandas=False)
    # Generate videos for each episode
    n = 0
    for episode_id, episode_data in enumerate(data[::-1]):
        positions = episode_data["episode/positions"]
        create_video_from_positions(cross_str, 0, 1, positions, episode_id)
        n += 1
        if n >= 3:
            break


if __name__ == "__main__":
    make_cross_movie()
