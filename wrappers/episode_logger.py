from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType


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
        if terminated or truncated:
            self.logger.log(
                {
                    "episode/length": self.length,
                    "episode/reward": self.reward,
                }
            )
            self.length = 0
            self.reward = 0
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        retv = self.env.reset(**kwargs)
        self.n_episodes += 1
        self.logger.log(
            {
                "episode": self.n_episodes,
            }
        )
        return retv
