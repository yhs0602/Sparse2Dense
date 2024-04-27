import argparse
import os.path
import random

import gymnasium
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from cross.cross_env import CROSS_GOALS, make_cross_env
from sb3_exts.episode_start_callback import EpisodeStartCallback
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.dense_maze_wrapper import DenseMazeWrapper
from wrappers.episode_logger import EpisodeLoggerWrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper
from wrappers.maze_selection_wrapper import MazeSelectionWrapper
from wrappers.position_logger import PositionLoggingWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper

current_path = os.path.dirname(os.path.abspath(__file__))
map_path = os.path.join(current_path, "hmaze1_colored.nbt")

# 실험 설명
# 학습할 때는 저 Goals 중 두 개를 랜덤하게 선택해서 학습합니다.
# 학습이 끝나면 3개의 Goals에 대해 전부 테스트합니다.

TEST_GOAL_IDX = 2
TRAIN_GOALS = [goal for i, goal in enumerate(CROSS_GOALS) if i != TEST_GOAL_IDX]
TEST_GOAL = CROSS_GOALS[TEST_GOAL_IDX]


def select_goal():
    return random.choice(TRAIN_GOALS)


eval_idx = 0


def select_goal_eval():
    global eval_idx

    goal = CROSS_GOALS[eval_idx % 3]
    eval_idx += 1
    return goal


def wrap_env(
    env, size_x, size_y, central_logger, goal_selector, is_eval: bool
) -> gymnasium.Env:
    return LogFlushWrapper(
        FastResetWrapper(
            EpisodeLoggerWrapper(
                # Truncate the episode if it takes too long
                TimeLimit(
                    # Living penalty
                    LivingPenaltyWrapper(
                        # Dense reward
                        DenseMazeWrapper(
                            SparseRewardWrapper(
                                # Checks, Logs, Terminates
                                MazeReachCheckAndLogWrapper(
                                    # Select goal when reset
                                    MazeSelectionWrapper(
                                        PositionLoggingWrapper(
                                            Turn90Wrapper(
                                                VisionWrapper(
                                                    env,
                                                    x_dim=size_x,
                                                    y_dim=size_y,
                                                ),
                                            ),
                                            logger=central_logger,
                                        ),
                                        goal_selector=goal_selector,
                                    ),
                                    radius=2,
                                    central_logger=central_logger,
                                    cooldown=2,
                                ),
                                reward=1,
                            ),
                            radius=5,
                            reward=0.001,
                        ),
                        penalty_abs=0.0001,
                    ),
                    max_episode_steps=20000,
                ),
                logger=central_logger,
            )
        ),
        logger=central_logger,
        is_eval=is_eval,
    )


def generalized_refactored_hmaze(
    port1: int = 8001, port2: int = 8002, device_id: int = 0
):
    group_name = f"v1-cross-dense-{TEST_GOAL_IDX}"
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
    central_logger = CentralLogger()
    for goal in CROSS_GOALS:
        wandb.define_metric(
            f"{goal}/success_count", summary="max", step_metric="episode"
        )
        wandb.define_metric(f"{goal}/time_took", step_metric="episode")
        wandb.define_metric(
            f"eval_{goal}/success_count", summary="max", step_metric="episode"
        )
        wandb.define_metric(f"eval_{goal}/time_took", step_metric="episode")
    wandb.define_metric("eval_episode/length", summary="max", step_metric="episode")
    wandb.define_metric("eval_episode/reward", summary="max", step_metric="episode")
    size_x = 114
    size_y = 64

    # Setup train environment
    base_env, _ = make_cross_env(port1, size_x, size_y)
    env = wrap_env(base_env, size_x, size_y, central_logger, select_goal, is_eval=False)
    env = DummyVecEnv([lambda: env])

    # Setup eval environment
    eval_base_env, _ = make_cross_env(port2, size_x, size_y)
    eval_env = wrap_env(
        eval_base_env, size_x, size_y, central_logger, select_goal_eval, is_eval=True
    )
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
        eval_freq=100,
        n_eval_episodes=6,
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
                # EpisodeLogger(),
                EpisodeStartCallback(eval_callback),
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
    TRAIN_GOALS = [goal for i, goal in enumerate(CROSS_GOALS) if i != TEST_GOAL_IDX]
    TEST_GOAL = CROSS_GOALS[TEST_GOAL_IDX]
    port1 = args.port1
    port2 = args.port2
    device_id = args.device_id
    generalized_refactored_hmaze(port1=port1, port2=port2, device_id=device_id)
