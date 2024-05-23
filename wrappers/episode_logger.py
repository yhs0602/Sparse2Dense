from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType

from cross_w2.cross_w2_env import Goal


class EpisodeLoggerWrapper(gymnasium.Wrapper):
    def __init__(self, env, logger, **kwargs):
        super().__init__(env)
        self.logger = logger
        self.n_episodes = 0
        self.length = 0
        self.reward = 0

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.length += 1
        self.reward += reward
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        goal: Goal = self.get_wrapper_attr("maze_goal")
        retv = self.env.reset(**kwargs)
        self.logger.log(
            {
                "episode/length": self.length,
                "episode/reward": self.reward,
                "episode/goal_idx": goal.idx if goal else -1,
                "episode": self.n_episodes,
            }
        )
        self.length = 0
        self.reward = 0
        self.n_episodes += 1
        return retv
