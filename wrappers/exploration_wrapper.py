from typing import SupportsFloat, Any, Optional, Tuple

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper


# Satisfy PBRS assumptions.
# Reward for moving away from origin, reduce reward for moving closer.


class ExplorationWrapper(Wrapper):
    def __init__(
        self,
        env,
        origin: Tuple[float, float, float],
        reward: float,
        **kwargs,
    ):
        self.env = env
        self.reward = reward
        self.origin = origin
        self.previous_distance = 0
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        x = info_obs.x
        y = info_obs.y
        z = info_obs.z
        new_distance = self.taxicab_distance(x, y, z)
        if new_distance > self.previous_distance:
            reward += self.reward
        elif new_distance < self.previous_distance:
            reward -= self.reward
        self.previous_distance = new_distance

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
        self.previous_distance = 0
        return obs, info

    def taxicab_distance(self, x: float, y: float, z: float) -> float:
        return (
            abs(x - self.origin[0]) + abs(y - self.origin[1]) + abs(z - self.origin[2])
        )
