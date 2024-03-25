from typing import SupportsFloat, Any, Optional, Tuple

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper


class MazeSuccessWrapper(Wrapper):
    def __init__(
        self,
        env,
        goal: Tuple[float, float, float],
        radius: float,
        reward: float,
        **kwargs,
    ):
        self.env = env
        self.goal = goal
        self.radius = radius
        self.reward = reward
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        x = info_obs.x
        y = info_obs.y
        z = info_obs.z

        # square goal check
        if (
            self.goal[0] - self.radius <= x <= self.goal[0] + self.radius
            and self.goal[1] - self.radius <= y <= self.goal[1] + self.radius
            and self.goal[2] - self.radius <= z <= self.goal[2] + self.radius
        ):
            reward += self.reward
            print("Goal Reached")
            terminated = True

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
