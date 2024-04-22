import random
import sys
import time

import numpy as np
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

from h_maze.h_maze_env import H_MAZE_GOALS, make_h_maze_env
from utils.central_logger import CentralLogger
from wrappers.episode_logger import EpisodeLoggerWrapper
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper
from wrappers.maze_selection_wrapper import MazeSelectionWrapper
from wrappers.position_logger import PositionLoggingWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper


def select_goal():
    return random.choice(H_MAZE_GOALS)


def h_maze_random():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="h-random",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    central_logger = CentralLogger()
    for goal in H_MAZE_GOALS:
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
    base_env, _ = make_h_maze_env(port=8001, size_x=size_x, size_y=size_y)
    env = LogFlushWrapper(
        FastResetWrapper(
            EpisodeLoggerWrapper(
                TimeLimit(
                    LivingPenaltyWrapper(
                        SparseRewardWrapper(
                            MazeReachCheckAndLogWrapper(
                                MazeSelectionWrapper(
                                    PositionLoggingWrapper(
                                        Turn90Wrapper(
                                            VisionWrapper(
                                                base_env,
                                                x_dim=size_x,
                                                y_dim=size_y,
                                            )
                                        ),
                                        logger=central_logger,
                                    ),
                                    goal_selector=select_goal,
                                ),
                                radius=2,
                                central_logger=central_logger,
                            ),
                            reward=1,
                        ),
                        penalty_abs=0.0001,
                    ),
                    max_episode_steps=20000,
                ),
                logger=central_logger,
            ),
        ),
        logger=central_logger,
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 40000 == 0,
        video_length=20000,
    )

    try:
        vec_env = env
        obs = vec_env.reset()
        start_time = time.time_ns()
        for i in range(9000000):
            # sample one from the action space
            action = random.sample([0, 1, 2], 1)
            action = np.array(action)
            # print(f"Action: {action}")
            obs, reward, done, info = vec_env.step(action)
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
            if i % 4000 == 0:
                print(f"Step: {i}")
        run.finish()
    finally:
        base_env.terminate()


if __name__ == "__main__":
    h_maze_random()
