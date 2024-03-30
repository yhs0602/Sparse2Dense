import os.path
import random

import numpy as np
import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from wrappers.maze_success_wrapper import MazeSuccessWrapper

GOALS = [
    (21, 1, 1),  # 앞쪽
    (21, 1, 14),  # 앞 오른쪽
    (3, 1, 14),  # 뒤 오른쪽
]


def select_goal():
    return random.choice(GOALS)


def structure_any():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="hmaze-noreward-random-goal",
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
            port=8001,
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
                "place template minecraft:hmaze1 0 0 0",
                "tp @p 3 1 1 -90 0",
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=False,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                os.path.abspath("hmaze1.nbt"),
            ],
        ),
        [],
    )
    env = FastResetWrapper(
        MazeSuccessWrapper(
            ActionWrapper(
                VisionWrapper(
                    base_env,
                    x_dim=size_x,
                    y_dim=size_y,
                ),
                enabled_actions=[
                    Action.FORWARD,
                    Action.TURN_LEFT,
                    Action.TURN_RIGHT,
                ],
            ),
            goal_selector=select_goal,
            reward=0,
            radius=2,
        ),
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 4000 == 0,
        video_length=400,
    )

    if False:
        model = A2C(
            "CnnPolicy", env, verbose=1, device="mps", tensorboard_log=f"runs/{run.id}"
        )

        model.learn(
            total_timesteps=300000,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
        )
        # model.save("dqn_sound_husk")
    else:
        vec_env = env
        obs = vec_env.reset()
        for i in range(900000):
            # sample one from the action space
            action = random.sample([0, 1, 2], 1)
            action = np.array(action)
            # print(f"Action: {action}")
            obs, reward, done, info = vec_env.step(action)
            if i % 4000 == 0:
                print(f"Step: {i}")
    run.finish()
    base_env.terminate()


if __name__ == "__main__":
    structure_any()
