import argparse
import os
from typing import Tuple, Any, Optional, Dict, SupportsFloat

import gymnasium
import torch as th
import wandb
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.core import WrapperObsType, WrapperActType
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from room.room_env import (
    select_goal_spawn,
    spawn_goal_command,
    remove_goal_command,
    make_room_env,
)
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.turn_90_wrapper import Turn90Wrapper


def patched_get_distribution(
    self,
    obs: th.Tensor,
    lstm_states: Tuple[th.Tensor, th.Tensor],
    episode_starts: th.Tensor,
) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
    """
    Get the current policy distribution given the observations.

    :param obs: Observation.
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :return: the action distribution and new hidden states.
    """
    # Call the method from the parent of the parent class
    features = super(ActorCriticPolicy, self).extract_features(
        obs, self.pi_features_extractor
    )
    latent_pi, lstm_states = self._process_sequence(
        features, lstm_states, episode_starts, self.lstm_actor
    )
    latent_pi = self.mlp_extractor.forward_actor(latent_pi)
    # Log (RGB, features, latent_pi, action, position)
    # RGB = obs
    # features = features
    # latent_pi = latent_pi
    # action = ?
    # position = ?
    wandb.log(
        {
            "RGB": obs,
            "features": features,
            "latent_pi": latent_pi,
        },
        commit=False,
    )
    return self._get_action_dist_from_latent(latent_pi), lstm_states


def patched__predict(
    self,
    observation: th.Tensor,
    lstm_states: Tuple[th.Tensor, th.Tensor],
    episode_starts: th.Tensor,
    deterministic: bool = False,
) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
    """
    Get the action according to the policy for a given observation.

    :param observation:
    :param lstm_states: The last hidden and memory states for the LSTM.
    :param episode_starts: Whether the observations correspond to new episodes
        or not (we reset the lstm states in that case).
    :param deterministic: Whether to use stochastic or deterministic actions
    :return: Taken action according to the policy and hidden states of the RNN
    """
    distribution, lstm_states = self.get_distribution(
        observation, lstm_states, episode_starts
    )
    actions = distribution.get_actions(deterministic=deterministic)
    # actions = actions
    wandb.log({"actions": actions}, commit=False)
    return actions, lstm_states


class GetPositionWrapper(gymnasium.Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info_obs = info["obs"]
        pos = (info_obs.x, info_obs.y, info_obs.z, info_obs.yaw)
        wandb.log(
            {"position_x": pos[0], "position_z": pos[2], "position_yaw": pos[3]},
            commit=False,
        )
        return obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        pos = (info_obs.x, info_obs.y, info_obs.z, info_obs.yaw)
        wandb.log(
            {"position_x": pos[0], "position_z": pos[2], "position_yaw": pos[3]},
            commit=False,
        )
        return obs, reward, terminated, truncated, info


def wrap_env(env, size_x, size_y, central_logger: CentralLogger) -> gymnasium.Env:
    return FastResetWrapper(
        # Truncate the episode if it takes too long
        TimeLimit(
            # Checks, Logs, Terminates
            RoomReachCheckAndLogWrapper(
                # Select goal when reset
                RoomGoalSelectionWrapper(
                    GetPositionWrapper(
                        Turn90Wrapper(
                            VisionWrapper(
                                env,
                                x_dim=size_x,
                                y_dim=size_y,
                            ),
                        )
                    ),
                    logger=central_logger,
                    goal_selector=select_goal_spawn,
                    goal_set_command_provider=spawn_goal_command,
                    goal_remove_command_provider=remove_goal_command,
                ),
                radius=2,
                central_logger=central_logger,
                cooldown=2,
            ),
            max_episode_steps=20000,
        ),
        logger=central_logger,
    )


def main(checkpoint_path: str, port1: int, device_id: int):
    checkpoint_name = os.path.basename(checkpoint_path)
    group_name = f"viz-{checkpoint_name}"
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="journal-viz",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group_name,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    size_x = 114
    size_y = 64

    # Setup train environment
    base_env, _ = make_room_env(
        port1,
        size_x,
        size_y,
        verbose=False,
        verbose_python=False,
        verbose_gradle=False,
        verbose_jvm=False,
    )
    central_logger = CentralLogger()
    base_env = wrap_env(base_env, size_x, size_y, central_logger)

    env = DummyVecEnv([lambda: base_env])
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
        device=get_device(device_id),
        tensorboard_log=f"runs/{run.id}",
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    if os.path.exists(checkpoint_path):
        model.load(checkpoint_path)
    else:
        raise FileNotFoundError(f"Context file {checkpoint_path} not found")
    # Patch RecurrentActorCriticPolicy.get_distribution
    # Rollout
    RecurrentActorCriticPolicy.get_distribution = patched_get_distribution
    RecurrentActorCriticPolicy._predict = patched__predict

    # Rollout
    try:
        obs = env.reset()
        _state = None
        for i in range(20000):
            wandb.log({"step": i}, commit=True)
            action, _state = model.predict(obs, deterministic=False, state=_state)
            obs, reward, done, info = env.step(action)
            if done:
                print(f"Done at {i}")
                break
    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8001, help="Port for rollout")
    arg_parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    args = arg_parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        port1=args.port,
        device_id=args.device_id,
    )
