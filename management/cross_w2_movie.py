# DEPRECATED
# 13 x 13

import wandb

from management.make_h_movie import create_video_from_positions
from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

cross_str = [
    "xxxxxxooxxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xoooooxxooooox",
    "oxxxxxxxxxxxxo",
    "oxxxxxxxxxxxxo",
    "xoooooxxooooox",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxxooxxxxxx",
]

assert len(cross_str) == 14


# start point: 2 32 6 > 0 0 0


def make_cross_w2_movie():
    # Initialize WandB API
    api = wandb.Api(timeout=120)
    run_names = [
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/ut8b91hh",  # Sparse 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/0ps4e1nn",  # Dense 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/jhhmgo0f"  # 3M 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/ktxlocs3",  # 2M 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/u0bwff7b",  # Sparse 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/0itxorpm",  # Dense 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/nk54yfnh",  # 3M 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/edsz2myn",  # 2M 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/3r4ctsay",  # Sparse 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/8ilqwrp5",  # Dense 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/wocxlma6",  # 3M 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/yxbvuc6o",  # 2M 2
    ]
    for run_name in run_names:
        run = api.run(run_name)

        # Get the data
        data = run.history(keys=["episode/positions"], pandas=False)
        # Generate the video for each episode
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            create_video_from_positions(cross_str, 0, 1, positions, episode_id)
            n += 1
            # Take the last 3 episodes, for now
            if n >= 3:
                break


if __name__ == "__main__":
    make_cross_w2_movie()
