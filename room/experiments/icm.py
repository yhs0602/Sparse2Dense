import argparse
import os
from typing import Optional

import gymnasium
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from rllte.xplore.reward import ICM, NGU, E3B
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from room.experiments.transpose import VisionTransposeWrapper
from room.room_env import (
    select_goal_spawn,
    define_room_metrics,
    spawn_goal_command,
    remove_goal_command,
    make_room_env,
)
from room.wrappers.room_episode_logger import RoomEpisodeLoggerWrapper
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper
from sb3_exts.custom_checkpoint_callback import CustomCheckpointCallback
from sb3_exts.rlex_callback import RLeXploreWithOnPolicyRL

# from sb3_exts.episode_start_callback import EpisodeStartCallback
from sb3_exts.rlex_pbim_callback import RLeXplorePBIMWithOnPolicyRL
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wandb_envs import WANDB_PROJECT, WANDB_ENTITY
from wrappers.living_penalty import LivingPenaltyWrapper
from wrappers.log_flush_wrapper import LogFlushWrapper
from wrappers.position_logger import PositionLoggingWrapper
from wrappers.sparse_maze_wrapper import SparseRewardWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper


# then the training will randomly start in those four areas.
# and the goals will also occur within those zones.
# I'll log the starting coordinates and which one of the four it is.


def wrap_env(
    env,
    size_x,
    size_y,
    central_logger,
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
    env = VisionTransposeWrapper(x_dim=size_x, y_dim=size_y, env=maze_wrapper)
    return LogFlushWrapper(
        FastResetWrapper(
            RoomEpisodeLoggerWrapper(
                # Truncate the episode if it takes too long
                TimeLimit(
                    # Living penalty
                    LivingPenaltyWrapper(
                        # Sparse to Dense reward
                        env=SparseRewardWrapper(
                            env,
                            reward=1,
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


def icm_transition(
    port1: int = 8001,
    device_id: int = 0,
    transition_timing: int = 3000000,
    extended: bool = False,
    max_steps: int = 10000000,
    seed: int = 3,
    base_checkpoint: Optional[str] = None,
    entropy_coef: float = 0.005,
    ir_type: str = "icm",
    ir_scale: float = 0.01,
    use_pbim: bool = False,
):
    set_random_seed(seed)
    from_str = ""
    if base_checkpoint:
        if "sparse" in base_checkpoint:
            from_str = "sparse"
        elif "dense" in base_checkpoint:
            from_str = "dense"
    # setting = select_goal_spawn()
    if use_pbim:
        group_name = f"v40-room-s2d-{ir_type}-from_{from_str}-{entropy_coef}-ir{ir_scale}-pbim"  # {setting['spawn_idx']}
    else:
        group_name = f"v40-room-s2d-{ir_type}-from_{from_str}-{entropy_coef}-ir{ir_scale}"  # {setting['spawn_idx']}
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        tags=["room-v1"],
        config={
            "seed": seed,
            "entropy_coef": entropy_coef,
            "extended": extended,
            "from_str": from_str,
        },
    )
    central_logger = CentralLogger()
    define_room_metrics()
    size_x = 114
    size_y = 64

    device = get_device(device_id)

    # Setup train environment
    base_env, _ = make_room_env(
        port1, size_x, size_y, extended=extended, verbose_gradle=True, verbose_jvm=True
    )
    env = wrap_env(
        base_env,
        size_x,
        size_y,
        central_logger,
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    if ir_type == "icm":
        irs = ICM(env, str(device))
    elif ir_type == "ngu":
        irs = NGU(env, str(device), mrs=ir_scale)
        ir_scale = 1
    elif ir_type == "e3b":
        irs = E3B(env, str(device))
        ir_scale = 1
    else:
        raise ValueError(f"Unknown intrinsic reward type: {ir_type}")
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 20000 == 0,
        video_length=20000,
    )
    # Setup eval environment
    # eval_base_env, _ = make_room_env(port2, size_x, size_y, verbose_gradle=True)
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

    if base_checkpoint:
        if os.path.exists(base_checkpoint):
            model = RecurrentPPO.load(base_checkpoint, env=env, device=device)
            print(f"Loaded checkpoint {base_checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint {base_checkpoint} not found")
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            env,
            verbose=1,
            device=device,
            tensorboard_log=f"runs/{run.id}",
            gae_lambda=0.99,
            ent_coef=entropy_coef,
            n_steps=512,
        )
        print("Using fresh model")

    checkpoint_steps = [
        1000000,
        2000000,
        2500000,
        3000000,
        3500000,
        4000000,
        5000000,
        6000000,
        7000000,
        8000000,
    ]
    checkpoint_callback = CustomCheckpointCallback(
        steps=checkpoint_steps,
        save_path=f"models/{extended}/ir-{ir_type}/{seed}",
        verbose=1,
    )

    if use_pbim:
        rlx_callback = RLeXplorePBIMWithOnPolicyRL(irs, ir_scale)
    else:
        rlx_callback = RLeXploreWithOnPolicyRL(irs, ir_scale)

    try:
        model.learn(
            total_timesteps=max_steps,
            callback=[
                WandbCallback(
                    gradient_save_freq=500,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ),
                rlx_callback,
                checkpoint_callback,
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
    arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--goal", type=int, default=2, help="Goal index to test")
    arg_parser.add_argument("--port1", type=int, default=8001, help="Port for training")
    # arg_parser.add_argument("--port2", type=int, default=8002, help="Port for testing")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    # arg_parser.add_argument(
    #     "--transition-timing",
    #     type=int,
    #     default=250,
    #     help="Reward transition timing in timesteps S->D; 10_000_000; 2000000, 3000000, 4000000",
    # )
    arg_parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    arg_parser.add_argument(
        "--extended",
        default=False,
        action="store_true",
        help="Use extended room environment",
    )
    arg_parser.add_argument(
        "--max-steps",
        type=int,
        default=1000_0000,
        help="Maximum number of steps to train the model for",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=3,
        help="Random seed",
    )
    arg_parser.add_argument(
        "--base-checkpoint",
        type=str,
        help="Base checkpoint to resume from",
        default=None,
    )
    arg_parser.add_argument(
        "--entropy",
        type=float,
        help="Entropy coefficient",
        default=0.005,
    )
    arg_parser.add_argument(
        "--ir-type",
        type=str,
        help="Intrinsic reward type",
        default="icm",
        choices=["icm", "ngu", "e3b"],
    )
    arg_parser.add_argument(
        "--ir-scale",
        type=float,
        help="Intrinsic reward scale",
        default=0.01,
    )
    arg_parser.add_argument(
        "--pbim",
        action="store_true",
        help="Use PBIM",
    )

    args = arg_parser.parse_args()
    port1 = args.port1
    # port2 = args.port2
    device_id = args.device_id

    icm_transition(
        port1=port1,
        device_id=device_id,
        extended=args.extended,
        max_steps=args.max_steps,
        seed=args.seed,
        base_checkpoint=args.base_checkpoint,
        entropy_coef=args.entropy,
        ir_type=args.ir_type,
        ir_scale=args.ir_scale,
        use_pbim=args.pbim,
    )
