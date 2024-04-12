import os.path
import random
from typing import Tuple

import gymnasium
import wandb
from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from craftground.wrappers.vision import VisionWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from check_vglrun import check_vglrun
from get_device import get_device
from h_maze.episode_reward_logger import EpisodeLogger
from h_maze.turn_90_wrapper import Turn90Wrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper
from wrappers.maze_selection_wrapper import MazeSelectionWrapper
from wrappers.sparse_maze_wrapper import SparseMazeWrapper

import argparse

current_path = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(current_path, "hmaze1_colored.nbt")

GROUND_GOALS = [
    (21, 1, 1),  # 앞쪽
    (21, 1, 14),  # 앞 오른쪽
    (3, 1, 14),  # 뒤 오른쪽
]

# 실험 설명
# 학습할 때는 저 Goals 중 두 개를 랜덤하게 선택해서 학습합니다.
# 학습이 끝나면 나머지 하나의 Goal을 선택해서 테스트합니다.
# 근데 랜덤하게 한다기보다는 어차피 두 개를 학습하고 나머지 하나를 테스트하는 것이므로
# 3개의 버전을 만들어서 각각 다른 Goal을 학습하고 테스트하도록 합니다.

TEST_GOAL_IDX = 2
TRAIN_GOALS = [goal for i, goal in enumerate(GROUND_GOALS) if i != TEST_GOAL_IDX]
TEST_GOAL = GROUND_GOALS[TEST_GOAL_IDX]


def select_goal():
    return random.choice(TRAIN_GOALS)


def make_env(
    port: int, size_x: int, size_y: int
) -> Tuple[CraftGroundEnvironment, list[str]]:
    return (
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


def wrap_env(env, size_x, size_y, goal_selector) -> gymnasium.Env:
    return FastResetWrapper(
        # Truncate the episode if it takes too long
        TimeLimitWrapper(
            # Living penalty
            LivingPenaltyWrapper(
                # Sparse reward
                SparseMazeWrapper(
                    # Checks, Logs, Terminates
                    MazeReachCheckAndLogWrapper(
                        # Select goal when reset
                        MazeSelectionWrapper(
                            Turn90Wrapper(
                                VisionWrapper(
                                    env,
                                    x_dim=size_x,
                                    y_dim=size_y,
                                ),
                            ),
                            goal_selector=goal_selector,
                        ),
                        radius=2,
                    ),
                    reward=1,
                ),
                penalty_abs=0.0001,
            ),
            max_timesteps=20000,
        ),
    )


def generalized_refactored_hmaze(
    port1: int = 8001, port2: int = 8002, device_id: int = 0
):
    group_name = f"hcrmaze-generalization{TEST_GOAL_IDX}"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    for goal in GROUND_GOALS:
        wandb.define_metric(f"{goal}/success_count", summary="max")
        wandb.define_metric(f"{goal}/time_took", step_metric=f"{goal}/success_count")
    size_x = 114
    size_y = 64

    # Setup train environment
    base_env, _ = make_env(port1, size_x, size_y)
    env = wrap_env(base_env, size_x, size_y, select_goal)
    env = DummyVecEnv([lambda: env])

    # Setup eval environment
    eval_base_env, _ = make_env(port2, size_x, size_y)
    eval_env = wrap_env(eval_base_env, size_x, size_y, lambda: TEST_GOAL)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = Monitor(eval_env)
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=20000,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run.id}",
        log_path=f"logs/{run.id}",
        eval_freq=400_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        device=get_device(device_id),
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                EpisodeLogger(),
                eval_callback,
            ],
        )
        model.save(f"{group_name}.ckpt")

        run.finish()
    finally:
        base_env.terminate()
        eval_base_env.terminate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--goal", type=int, default=2, help="Goal index to test")
    arg_parser.add_argument("--port1", type=int, default=8001, help="Port for training")
    arg_parser.add_argument("--port2", type=int, default=8002, help="Port for testing")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    args = arg_parser.parse_args()
    TEST_GOAL_IDX = args.goal
    TRAIN_GOALS = [goal for i, goal in enumerate(GROUND_GOALS) if i != TEST_GOAL_IDX]
    TEST_GOAL = GROUND_GOALS[TEST_GOAL_IDX]
    port1 = args.port1
    port2 = args.port2
    device_id = args.device_id
    generalized_refactored_hmaze(port1=port1, port2=port2, device_id=device_id)
