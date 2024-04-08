import os.path
import random

import wandb
from craftground import craftground
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from craftground.wrappers.vision import VisionWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from check_vglrun import check_vglrun
from get_device import get_device
from h_maze.episode_reward_logger import EpisodeLogger
from h_maze.turn_90_wrapper import Turn90Wrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.maze_success_wrapper import MazeSuccessWrapper

current_path = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(current_path, "hmaze1_colored.nbt")

GOALS = [
    (21, 1, 1),  # 앞쪽
    (21, 1, 14),  # 앞 오른쪽
    (3, 1, 14),  # 뒤 오른쪽
]


def select_goal():
    return random.choice(GOALS)


def hmaze_rppo_sparse():
    group_name = "hcrmaze-gae0.99-ent0.005step512-sparse"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group_name,
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
                "time set noon",
                "place template minecraft:hmaze1_colored 0 0 0",
                "tp @p 3 1 1 -90 0",
                "effect give @p minecraft:speed infinite 2 true",  # speed effect, particle hidden
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
                MazeSuccessWrapper(
                    Turn90Wrapper(
                        VisionWrapper(
                            base_env,
                            x_dim=size_x,
                            y_dim=size_y,
                        ),
                    ),
                    goal_selector=select_goal,
                    reward=1,
                    radius=2,
                ),
                penalty_abs=0.0001,
            ),
            max_timesteps=20000,
        ),
    )
    env = DummyVecEnv([lambda: env])
    env = Monitor(env)
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=20000,
    )

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        device=get_device(),
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    try:
        model.learn(
            total_timesteps=6000000,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                EpisodeLogger(),
            ],
        )
        model.save(f"{group_name}.ckpt")
        run.finish()
    finally:
        base_env.terminate()


if __name__ == "__main__":
    hmaze_rppo_sparse()
