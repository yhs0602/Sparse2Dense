from typing import List, SupportsFloat, Any, Tuple, Dict

import gymnasium
from gymnasium import Env
from gymnasium.core import WrapperActType, WrapperObsType


class RewardTransitionWrapper(gymnasium.Wrapper):
    """
    A wrapper that modifies the reward based on the episode count.
    :param env: The environment
    :param reward_envs: A list of reward environments to apply to the environment
    (observation, reward, terminated, truncated, info) -> reward
    :param transition_timings: A list of timings in total_timesteps to transition between reward environments
    len(transition_timings) == len(reward_envs) - 1
    The episode number starts from 1
    For example, if reward_envs = [env1, env2, env3] and transition_timings = [10, 20],
    the reward environment will be env1 for episodes 1-10, env2 for episodes 11-20, and env3 for episodes 21 onwards
    """

    def __init__(self, reward_envs: List[Env], transition_timings: List[int], **kwargs):
        super().__init__(reward_envs[0])
        assert len(reward_envs) - 1 == len(transition_timings)
        self.episode_count = 0
        self.reward_envs = reward_envs
        self.transition_timings = transition_timings
        self.env_idx = 0
        self.total_timesteps = 0

    def step(
        self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.total_timesteps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.episode_count += 1
        if self.env_idx < len(self.transition_timings):
            if self.total_timesteps >= self.transition_timings[self.env_idx]:
                self.env_idx += 1
        self.env = self.reward_envs[self.env_idx]
        return self.env.reset(**kwargs)
