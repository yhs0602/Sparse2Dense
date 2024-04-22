import argparse
import os.path
import random
import sys
import time

import numpy as np
import wandb
from craftground import craftground
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from check_vglrun import check_vglrun
from cross.sparse_reward_wrapper import SparseRewardWrapper
from h_maze.turn_90_wrapper import Turn90Wrapper
from wrappers.living_penalty import LivingPenaltyWrapper

current_path = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(current_path, "cross.nbt")

INITIAL_POSITION = [2, 2, 6, -90, 0]

GOALS = [
    (7, 2, 1),  # 왼쪽
    (12, 2, 6),  # 앞쪽
    (7, 2, 11),  # 오른쪽
]


def select_goal():
    return random.choice(GOALS)


def cross_random(port: int):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="cross-random",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )
    for goal in GOALS:
        wandb.define_metric(f"{goal}/success_count", summary="max")
        wandb.define_metric(f"{goal}/time_took", step_metric=f"{goal}/success_count")
    size_x = 114
    size_y = 64
    base_env, sound_list = (
        craftground.make(
            port=port,
            initialInventoryCommands=[],
            verbose=False,
            initialPosition=[5, 5, 5],  # nullable
            initialMobsCommands=[],
            imageSizeX=size_x,
            imageSizeY=size_y,
            visibleSizeX=size_x,
            visibleSizeY=size_y,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=[],  # No sound subtitles
            miscStatKeys=[],  # No stats
            initialExtraCommands=[
                "time set noon",
                "place template minecraft:cross 0 0 0",
                "tp @p 2 2 6 -90 0",
                "effect give @p minecraft:speed infinite 1 true",  # speed effect, particle hidden
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=False,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                map_path,
            ],
            no_pov_effect=True,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            use_vglrun=check_vglrun(),
        ),
        [],
    )
    env = FastResetWrapper(
        TimeLimitWrapper(
            LivingPenaltyWrapper(
                SparseRewardWrapper(
                    CrossReachCheckAndLogWrapper(
                        Turn90Wrapper(
                            VisionWrapper(
                                base_env,
                                x_dim=size_x,
                                y_dim=size_y,
                            ),
                        ),
                        radius=2,
                    ),
                    reward=1,
                ),
                penalty_abs=0.0001,
            ),
            max_timesteps=15000,
        ),
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 400000 == 0,
        video_length=20000,
    )

    try:
        obs = env.reset()
        start_time = time.time_ns()
        for i in range(10_000_000):
            # sample one from the action space
            action = random.sample([0, 1, 2], 1)
            action = np.array(action)
            # print(f"Action: {action}")
            obs, reward, done, info = env.step(action)
            time_elapsed = max(
                (time.time_ns() - start_time) / 1e9, sys.float_info.epsilon
            )
            fps = int(i / time_elapsed)
            if i % 512 == 0:
                wandb.log(
                    {
                        "time/iterations": i,
                        "time/fps": fps,
                        "time/time_elapsed": int(time_elapsed),
                        "time/total_timesteps": i,
                    }
                )
            if i % 4096 == 0:
                print(f"Step: {i}")
        run.finish()
    finally:
        base_env.terminate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8001, help="Port for training")
    args = arg_parser.parse_args()
    cross_random(port=args.port)
