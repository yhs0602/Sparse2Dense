import argparse

import gymnasium
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO

# from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from room.room_env import (
    define_room_metrics,
    make_room_env,
    select_goal_spawn,
    spawn_goal_command,
    remove_goal_command,
)
from room.wrappers.room_dense_wrapper import HomeDenseWrapper
from room.wrappers.room_episode_logger import RoomEpisodeLoggerWrapper
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper

# from sb3_exts.episode_start_callback import EpisodeStartCallback
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.episode_logger import EpisodeLoggerWrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.position_logger import PositionLoggingWrapper
from wrappers.reward_transition import RewardTransitionWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper


# 실험 설명
# 학습할 때는 저 Goals 중 두 개를 랜덤하게 선택해서 학습합니다.
# 학습이 끝나면 3개의 Goals에 대해 전부 테스트합니다.


def wrap_env(
    env,
    size_x,
    size_y,
    central_logger,
    transition_timing: int,
) -> gymnasium.Env:
    # Checks, Logs, Terminates
    maze_wrapper = RoomReachCheckAndLogWrapper(
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
    )
    return LogFlushWrapper(
        FastResetWrapper(
            RoomEpisodeLoggerWrapper(
                # Truncate the episode if it takes too long
                TimeLimit(
                    # Living penalty
                    LivingPenaltyWrapper(
                        # Sparse to Dense reward
                        RewardTransitionWrapper(
                            reward_envs=[
                                HomeDenseWrapper(
                                    SparseRewardWrapper(
                                        maze_wrapper,
                                        reward=1,
                                    ),
                                    radius=5,
                                    reward=0.001,
                                ),
                                SparseRewardWrapper(
                                    maze_wrapper,
                                    reward=1,
                                ),
                            ],
                            transition_timings=[
                                transition_timing,
                            ],
                        ),
                        penalty_abs=0.0001,
                    ),
                    max_episode_steps=20000,
                ),
                logger=central_logger,
            )
        ),
        logger=central_logger,
    )


def room_d2s(
    port1: int,
    device_id: int,
    transition_timing: int,
):
    group_name = f"v31-room-v1-d2s-{transition_timing}"
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
    base_env, _ = make_room_env(port1, size_x, size_y)
    env = wrap_env(
        base_env,
        size_x,
        size_y,
        central_logger,
        transition_timing=transition_timing,
    )
    env = DummyVecEnv([lambda: env])
    env = Monitor(env)
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=20000,
    )
    # Setup eval environment
    # eval_base_env, _ = make_cross_w2_env(port2, size_x, size_y)
    # eval_env = wrap_env(
    #     eval_base_env,
    #     size_x,
    #     size_y,
    #     central_logger,
    #     select_goal_eval,
    #     is_eval=True,
    #     transition_timing=transition_timing,
    # )
    # eval_env = DummyVecEnv([lambda: eval_env])
    # eval_env = Monitor(eval_env)
    # eval_env = VecVideoRecorder(
    #     eval_env,
    #     f"videos/{run.id}",
    #     record_video_trigger=lambda x: x % 20000 == 0,
    #     video_length=20000,
    # )
    #
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
        model.save(f"ckpts/{group_name}-{run.name}.ckpt")

        run.finish()
    finally:
        base_env.terminate()
        # eval_base_env.terminate()


if __name__ == "__main__":
    # 10000000 steps = Almost 500+ episodes
    # Transition timing: 166, 250, 332
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--goal", type=int, default=2, help="Goal index to test")
    arg_parser.add_argument("--port1", type=int, default=8001, help="Port for training")
    # arg_parser.add_argument("--port2", type=int, default=8002, help="Port for testing")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    arg_parser.add_argument(
        "--transition-timing",
        type=int,
        default=250,
        help="Reward transition timing in timesteps S->D; 10_000_000; 2000000, 3000000, 4000000",
    )
    args = arg_parser.parse_args()
    port1 = args.port1
    # port2 = args.port2
    device_id = args.device_id
    transition_timing = args.transition_timing
    room_d2s(
        port1=port1,
        device_id=device_id,
        transition_timing=transition_timing,
    )
