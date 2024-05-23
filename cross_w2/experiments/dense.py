import argparse
import os
from typing import Optional

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

from cross_w2.cross_w2_env import (
    make_cross_w2_env,
    CROSS_W2_GOALS_INSTANCES,
)
from cross_w2.experiments.global_settings import (
    SUCCESS_RADIUS,
    SUCCESS_REWARD,
    DENSE_RADIUS,
    DENSE_REWARD,
    LIVING_PENALTY_ABS,
    MAX_EPISODE_TIMESTEPS,
    TOTAL_TIMESTEPS,
    EVAL_FREQ,
    EVAL_EPISODES,
)
from cross_w2.experiments.sparse import (
    TrainGoalSelector,
)
from define_metric import define_metrics
from sb3_exts.episode_start_callback import EpisodeStartCallback
from sb3_exts.last_checkpoint_callback import LastCheckpointCallback
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.changing_eval_wrapper import InjectedParameter, ChangingEvalWrapper
from wrappers.dense_maze_wrapper import DenseMazeWrapper
from wrappers.episode_logger import EpisodeLoggerWrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper
from wrappers.maze_selection_wrapper import MazeSelectionWrapper
from wrappers.position_logger import PositionAndRewardLoggingWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper


# 실험 설명
# 학습할 때는 저 Goals 중 두 개를 랜덤하게 선택해서 학습합니다.
# 학습이 끝나면 3개의 Goals에 대해 전부 테스트합니다.


def wrap_env(
    env, size_x, size_y, central_logger, omit_goal: int, is_eval: bool
) -> gymnasium.Env:
    train_goal_selector = TrainGoalSelector(omit_goal)
    # Shared among all train / eval
    basic_env = PositionAndRewardLoggingWrapper(
        Turn90Wrapper(
            VisionWrapper(
                env,
                x_dim=size_x,
                y_dim=size_y,
            ),
        ),
        logger=central_logger,
    )
    # Select goal when reset
    selection_env = MazeSelectionWrapper(basic_env)
    misc_env = LivingPenaltyWrapper(
        # Dense reward
        DenseMazeWrapper(
            SparseRewardWrapper(
                # Checks, Logs, Terminates
                MazeReachCheckAndLogWrapper(
                    selection_env,
                    radius=SUCCESS_RADIUS,
                    central_logger=central_logger,
                    cooldown=2,
                ),
                reward=SUCCESS_REWARD,
            ),
            radius=DENSE_RADIUS,
            reward=DENSE_REWARD,
        ),
        penalty_abs=LIVING_PENALTY_ABS,
    )
    # wrong_goal_penalty_env = WrongGoalPenaltyWrapper(
    #     env=misc_env,
    #     radius=PENALTY_RADIUS,
    #     central_logger=central_logger,
    #     cooldown=2,
    #     reward=WRONG_PENALTY,
    #     total_goals=CROSS_W2_GOALS_INSTANCES,
    # )
    env = LogFlushWrapper(
        FastResetWrapper(
            EpisodeLoggerWrapper(
                # Truncate the episode if it takes too long
                TimeLimit(
                    misc_env,
                    max_episode_steps=MAX_EPISODE_TIMESTEPS,
                ),
                logger=central_logger,
            )
        ),
        logger=central_logger,
        is_eval=is_eval,
    )
    if is_eval:
        # Provide how to select goal for evaluation, and other parameters
        eval_env = ChangingEvalWrapper(
            env,
            parameters=lambda reset_count: InjectedParameter(
                {
                    "goal": CROSS_W2_GOALS_INSTANCES[(reset_count - 1) % 3],
                    # 0, 1, 2, 0, 1, 2, ...
                    "enabled_earlystop": False,  # ((reset_count - 1) // 30) % 2 == 0,
                    # True * 30, False * 30, ...
                    "enabled_negative_reward": False,  # ((reset_count - 1) // 30) % 2 == 0,
                    # True * 30, False * 30, ...
                }
            ),
            period=60,
            central_logger=central_logger,
        )
        selection_env.variable_providers.append(eval_env)
        # wrong_goal_penalty_env.variable_providers.append(eval_env)
        return eval_env
    else:
        # Provide how to select goal for trainer
        selection_env.variable_providers.append(train_goal_selector)
        # wrong_goal_penalty_env.variable_providers.append(
        #     EnableWrongGoalPenaltyAndEarlyStopProvider()
        # )
        return env


def w2_maze_dense(
    port1: int = 8001,
    port2: int = 8002,
    device_id: int = 0,
    omit_goal_idx: int = 0,
    resume_id: Optional[str] = None,
    context_path: str = None,
):
    group_name = f"v13-crossw2-dense-{omit_goal_idx}"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        id=resume_id,  # resume_id가 있으면 해당 run을 resume합니다.
        resume="must" if resume_id else "never",
    )
    central_logger = CentralLogger()
    define_metrics(CROSS_W2_GOALS_INSTANCES)
    size_x = 114
    size_y = 64

    # Setup train environment
    base_env, _ = make_cross_w2_env(port1, size_x, size_y)
    env = wrap_env(
        base_env, size_x, size_y, central_logger, omit_goal_idx, is_eval=False
    )
    env = DummyVecEnv([lambda: env])

    # Setup eval environment
    eval_base_env, _ = make_cross_w2_env(port2, size_x, size_y)
    eval_env = wrap_env(
        eval_base_env, size_x, size_y, central_logger, omit_goal_idx, is_eval=True
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = Monitor(eval_env)
    eval_env = VecVideoRecorder(
        eval_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % MAX_EPISODE_TIMESTEPS == 0,
        video_length=MAX_EPISODE_TIMESTEPS,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{run.id}",
        log_path=f"logs/{run.id}",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=False,
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

    if resume_id and context_path:
        if os.path.exists(context_path):
            model.load(context_path)
        else:
            raise FileNotFoundError(f"Context file {context_path} not found")

    try:
        checkpoint_callback = LastCheckpointCallback(
            save_freq=100000,  # 약 18분 주기, 1%마다 # 100000
            save_path=f"mid_ckpts/{run.name}",  # wandb run 이름을 사용하여 저장 경로 설정
            name_prefix="model",
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                # EpisodeLogger(),
                EpisodeStartCallback(eval_callback),
                checkpoint_callback,
            ],
        )
        # To properly flush the logs
        env.reset()
        eval_env.reset()
        model.save(f"ckpts/{group_name}_{run.id}_final.ckpt.zip")

    finally:
        base_env.terminate()
        eval_base_env.terminate()
        run.finish()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--goal", type=int, default=2, help="Goal index to test")
    arg_parser.add_argument("--port1", type=int, default=8001, help="Port for training")
    arg_parser.add_argument("--port2", type=int, default=8002, help="Port for testing")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    arg_parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    arg_parser.add_argument("--resume-run-id", type=str, help="Run id to resume")
    arg_parser.add_argument("--context", type=str, help="Path to context file")
    args = arg_parser.parse_args()
    port1 = args.port1
    port2 = args.port2
    device_id = args.device_id
    w2_maze_dense(
        port1=port1,
        port2=port2,
        device_id=device_id,
        omit_goal_idx=args.goal,
        resume_id=args.resume_run_id,
        context_path=args.context,
    )
