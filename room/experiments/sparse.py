import argparse

import gymnasium
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO

# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from cross_w2.cross_w2_env import make_cross_w2_env
from room.room_env import (
    select_goal_spawn,
    define_room_metrics,
    spawn_goal_command,
    remove_goal_command,
)
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper

# from sb3_exts.episode_start_callback import EpisodeStartCallback
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.episode_logger import EpisodeLoggerWrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.position_logger import PositionLoggingWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper


# 그러면 트레이닝 시 저 4개의 영역에서 무작위로 시작하고
# 골도 저 영역안에서 생겨나되
# 4개중에 어디인지 랑 시작좌표 로깅해둘게요


def wrap_env(env, size_x, size_y, central_logger) -> gymnasium.Env:
    return LogFlushWrapper(
        FastResetWrapper(
            EpisodeLoggerWrapper(
                # Truncate the episode if it takes too long
                TimeLimit(
                    # Living penalty
                    LivingPenaltyWrapper(
                        # Sparse reward
                        SparseRewardWrapper(
                            # Checks, Logs, Terminates
                            RoomReachCheckAndLogWrapper(
                                # Select goal when reset
                                RoomGoalSelectionWrapper(
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
                                    goal_selector=select_goal_spawn,
                                    goal_set_command_provider=spawn_goal_command,
                                    goal_remove_command_provider=remove_goal_command,
                                ),
                                radius=2,
                                central_logger=central_logger,
                                cooldown=2,
                            ),
                            reward=1,
                        ),
                        penalty_abs=0.0001,
                    ),
                    max_episode_steps=20000,
                ),
                logger=central_logger,
                goal_key="goal",
            )
        ),
        logger=central_logger,
    )


def sparse_room(port1: int = 8001, device_id: int = 0):
    # setting = select_goal_spawn()
    group_name = f"v30-room-v1"  # {setting['spawn_idx']}
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        tags=["room-v1"],
    )
    central_logger = CentralLogger()
    define_room_metrics()
    size_x = 114
    size_y = 64

    # Setup train environment
    base_env, _ = make_cross_w2_env(port1, size_x, size_y)
    env = wrap_env(base_env, size_x, size_y, central_logger)
    env = DummyVecEnv([lambda: env])

    # Setup eval environment
    # eval_base_env, _ = make_cross_w2_env(port2, size_x, size_y, verbose_gradle=True)
    # eval_env = wrap_env(
    #     eval_base_env, size_x, size_y, central_logger, is_eval=True
    # )
    # eval_env = DummyVecEnv([lambda: eval_env])
    # eval_env = Monitor(eval_env)
    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 20000 == 0,
    #     video_length=20000,
    # )

    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=f"models/{run.id}",
    #     log_path=f"logs/{run.id}",
    #     eval_freq=500,
    #     n_eval_episodes=30,
    #     deterministic=False,
    #     render=False,
    # )

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
                # EpisodeStartCallback(eval_callback),
            ],
        )
        model.save(f"{group_name}.ckpt")

        run.finish()
    finally:
        base_env.terminate()
        # eval_base_env.terminate()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--goal", type=int, default=2, help="Goal index to test")
    arg_parser.add_argument("--port1", type=int, default=8001, help="Port for training")
    # arg_parser.add_argument("--port2", type=int, default=8002, help="Port for testing")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    arg_parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    args = arg_parser.parse_args()
    port1 = args.port1
    # port2 = args.port2
    device_id = args.device_id
    sparse_room(port1=port1, device_id=device_id)
