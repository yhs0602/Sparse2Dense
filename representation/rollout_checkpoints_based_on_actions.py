import argparse
import gzip
import json
import os
from dataclasses import dataclass
from typing import Tuple, Any, Optional, Dict, SupportsFloat, Union

import gymnasium
import numpy as np
import torch as th
from PIL import Image
from PIL.Image import Transpose
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.core import WrapperObsType, WrapperActType
from gymnasium.wrappers import TimeLimit
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

from room.room_env import (
    spawn_goal_command,
    remove_goal_command,
    make_room_env,
)
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper
from utils.central_logger import CentralLogger
from utils.get_device import get_device
from wrappers.turn_90_wrapper import Turn90Wrapper

spawn_x = 0
spawn_y = 0
spawn_z = 0
goal_x = 0
goal_y = 0
goal_z = 0

# Changes.
# image, pth separately → save as json when available
# Example: images / has images per run
# run_name.json in data.
# data as json (tensor as list)


@dataclass
class Row:
    features: th.Tensor
    latent_pi: th.Tensor
    actions: th.Tensor
    action: int
    position_x: float
    position_z: float
    position_yaw: float
    image_idx: int


def jsonize(the_dict: Dict):
    for key, value in the_dict.items():
        if isinstance(value, th.Tensor):
            the_dict[key] = value.tolist()
        if isinstance(value, np.ndarray):
            the_dict[key] = value.tolist()
    return the_dict


class Logger:
    def __init__(self, base_dir: str, run_name: str):
        self.base_dir = base_dir
        self.run_name = run_name
        self.data_dir = os.path.join(base_dir, "data")
        self.data_json_path = os.path.join(self.data_dir, f"{self.run_name}.json.gz")
        self.images_dir = os.path.join(self.base_dir, "images", self.run_name)
        # self.pth_name = os.path.join(self.data_dir, "data.pth")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        self.rows = []
        self.idx = 0
        self.buffer = {}
        self.image_buffer = None

    def log(self, data: Dict, commit: bool):
        self.buffer.update(data)
        if commit:
            data = Row(
                features=self.buffer["features"],
                latent_pi=self.buffer["latent_pi"],
                action=self.buffer["action"],
                actions=self.buffer["actions"],
                position_x=self.buffer["position_x"],
                position_z=self.buffer["position_z"],
                position_yaw=self.buffer["position_yaw"],
                image_idx=self.idx,
            )
            self.rows.append(data)
            self.buffer = {}

    def log_image(self, rgb: Optional[th.Tensor], commit: bool):
        if not commit:
            self.image_buffer = rgb
        else:
            image_path = os.path.join(self.images_dir, f"{self.idx:05d}.png")
            im = Image.fromarray(
                self.image_buffer.squeeze().permute(1, 2, 0).cpu().numpy(), mode="RGB"
            )
            im = im.transpose(
                Transpose.ROTATE_270
            )  # Rotate image 270 degrees to fit in normal orientation
            im.save(image_path)
            self.rows[-1].image_idx = self.idx
            self.idx += 1
            self.image_buffer = None

    def flush(self):
        # First flush json
        # [Row] -> json
        jsonized = [jsonize(row.__dict__) for row in self.rows]
        with gzip.open(self.data_json_path, "wt") as f:
            json.dump(jsonized, f)
        # torch.save(self.rows, self.pth_name)
        # Then flush images
        self.idx = 0


logger: Optional[Logger] = None


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
    logger.log(
        {
            "features": features,
            "latent_pi": latent_pi,
        },
        commit=False,
    )
    logger.log_image(obs, commit=False)
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
    logger.log({"actions": actions}, commit=False)
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
        logger.log(
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
        logger.log(
            {"position_x": pos[0], "position_z": pos[2], "position_yaw": pos[3]},
            commit=False,
        )
        return obs, reward, terminated, truncated, info


def fixed_select_goal_spawn() -> Dict[str, Union[int, Tuple[float, float, float]]]:
    return {
        "spawn_idx": 0,
        "spawn": (spawn_x, spawn_y, spawn_z),
        "goal": (goal_x, goal_y, goal_z),
    }


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
                    goal_selector=fixed_select_goal_spawn,
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


action_str_to_int = {
    "MOVE_FORWARD": 0,
    "TURN_LEFT_90": 1,
    "TURN_RIGHT_90": 2,
}

action_int_to_str = {
    0: "MOVE_FORWARD",
    1: "TURN_LEFT_90",
    2: "TURN_RIGHT_90",
}


def main(port1: int, device_id: int, trajectory_json: str, extended: bool):
    size_x = 114
    size_y = 64

    current_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = "0731_checkpoints"
    checkpoint_dir = os.path.join(current_path, checkpoint_dir)

    trajectory_json = os.path.join(current_path, trajectory_json)
    actions = json.loads(open(trajectory_json).read())["actions"]
    positions = json.loads(open(trajectory_json).read())["positions"]
    global spawn_x, spawn_y, spawn_z
    global goal_x, goal_y, goal_z
    spawn_x, spawn_y, spawn_z, _yaw = positions[0]

    # Answer: su80k2nq.csv.gz's 22 ; 3111
    # Goal = [8.741072837046506, 2, 17.98566927436925]
    # Length = 2647
    goal_x, goal_y, goal_z = 8.741072837046506, 2, 17.98566927436925

    print(f"{goal_x=} {goal_y=} {goal_z=} {spawn_x=} {spawn_y=} {spawn_z=}")

    print(f"{len(actions)=}")

    # Setup train environment
    base_env, _ = make_room_env(
        port1,
        size_x,
        size_y,
        extended=extended,
        verbose=False,
        verbose_python=False,
        verbose_gradle=False,
        verbose_jvm=False,
    )
    central_logger = CentralLogger()
    base_env = wrap_env(base_env, size_x, size_y, central_logger)

    env = DummyVecEnv([lambda: base_env])

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        verbose=1,
        device=get_device(device_id),
        gae_lambda=0.99,
        ent_coef=0.005,
        n_steps=512,
    )

    global logger
    try:
        for extended_dir in os.listdir(checkpoint_dir):  # extended, not_extended
            if extended_dir == "extended" and not extended:
                print(f"Skipping extended as {extended}")
                continue
            if extended_dir == "not_extended" and extended:
                print("Skipping not_extended")
                continue
            algos_dir = os.path.join(checkpoint_dir, extended_dir)
            for algo in os.listdir(algos_dir):  # dense, s2d, d2s, sparse
                algo_dir = os.path.join(algos_dir, algo)
                for run_name in os.listdir(algo_dir):  # abcdef
                    run_dir = os.path.join(algo_dir, run_name)
                    for checkpoint in os.listdir(run_dir):
                        checkpoint_path = os.path.join(run_dir, checkpoint)
                        if os.path.exists(checkpoint_path):
                            model = RecurrentPPO.load(checkpoint_path)
                            print(f"Loaded checkpoint {checkpoint_path}")
                        else:
                            raise FileNotFoundError(
                                f"Context file {checkpoint_path} not found"
                            )
                        base_dir = "./representation_data_intermediate_checkpoints"
                        log_dir = os.path.join(base_dir, extended_dir, algo)
                        logger = Logger(
                            log_dir,
                            f"{run_name}__{checkpoint}",
                        )
                        print(f"Created logger for {logger.run_name}")
                        # Patch RecurrentActorCriticPolicy.get_distribution
                        # Rollout
                        RecurrentActorCriticPolicy.get_distribution = (
                            patched_get_distribution
                        )
                        RecurrentActorCriticPolicy._predict = patched__predict

                        # Rollout
                        obs = env.reset()
                        _state = None
                        for i in range(len(actions)):
                            action, _state = model.predict(
                                obs, deterministic=False, state=_state
                            )
                            logger.log(
                                {
                                    "action": action,
                                    "action_str": action_int_to_str[int(action)],
                                    "step": i,
                                    "actual_action_str": actions[i],
                                    "actual_action_int": action_str_to_int[actions[i]],
                                },
                                commit=True,
                            )
                            logger.log_image(None, commit=True)
                            # Use fixed action, not the model's
                            action = actions[i]

                            obs, reward, done, info = env.step(
                                [action_str_to_int[action]]
                            )
                            if done:
                                print(f"Done at {i}!!!!!!!")
                        logger.flush()
                        # break
    finally:
        env.close()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--port", type=int, default=8001, help="Port for rollout")
    arg_parser.add_argument(
        "--device-id", type=int, default=0, help="CUDA Device ID for training"
    )
    arg_parser.add_argument(
        "--trajectory", type=str, default="selected_trajectory.json"
    )
    arg_parser.add_argument(
        "--extended",
        default=False,
        action="store_true",
        help="Use extended room environment",
    )
    args = arg_parser.parse_args()
    main(
        port1=args.port,
        device_id=args.device_id,
        trajectory_json=args.trajectory,
        extended=args.extended,
    )
