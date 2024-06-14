from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType


class RoomEpisodeLoggerWrapper(gymnasium.Wrapper):
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
            setting = self.get_wrapper_attr("room_settings")
            goal = setting["goal"]
            spawn_idx = setting["spawn_idx"]
            spawn = setting["spawn"]
            self.logger.log(
                {
                    "episode/length": self.length,
                    "episode/reward": self.reward,
                    "episode/goal": goal,
                    "episode/spawn": spawn,
                    "episode/spawn_idx": spawn_idx,
                    f"{spawn_idx}/time_took": self.length,
                    f"{spawn_idx}/reward": self.reward,
                    "episode/goal_x": goal[0],
                    "episode/goal_y": goal[1],
                    "episode/goal_z": goal[2],
                    "episode/spawn_x": spawn[0],
                    "episode/spawn_y": spawn[1],
                    "episode/spawn_z": spawn[2],
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
