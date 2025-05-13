from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper


# Give a dense reward to a certain distance from the goal point (within distance 5), end episode
# - Reward calculation
# - Reward calculation: Potential Based
# # - Use distance calculated by Taxicab distance
# - Reward per step: -0.01 for farther, 0.01 for closer
class HomeDenseWrapper(Wrapper):
    def __init__(
        self,
        env: Wrapper,
        radius: float,
        reward: float,
        **kwargs,
    ):
        self.env = env
        self.radius = radius
        self.reward = reward
        self.previous_squared_distance = radius
        self.settings = self.get_wrapper_attr("room_settings")
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["full"]
        x = info_obs.x
        y = info_obs.y
        z = info_obs.z
        # Dense goal check
        if self.within_range(x, y, z):  # Only when the agent is in the reward range
            new_squared_distance = self.squared_euclidian_distance(x, y, z)
            if new_squared_distance > self.previous_squared_distance:
                reward -= self.reward
            elif new_squared_distance < self.previous_squared_distance:
                reward += self.reward
            self.previous_squared_distance = new_squared_distance

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def within_range(self, x, y, z):
        goal = self.settings["goal"]
        return self.within_range_single(goal, x, y, z)

    def within_range_single(self, goal, x, y, z) -> bool:
        dist2 = (x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2
        return dist2 <= self.radius**2

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_squared_distance = self.radius
        return obs, info

    def squared_euclidian_distance(self, x: float, y: float, z: float) -> float:
        goal = self.settings["goal"]
        return self.squared_euclidian_distance_single(goal, x, y, z)

    def squared_euclidian_distance_single(
        self, goal, x: float, y: float, z: float
    ) -> float:
        return (x - goal[0]) ** 2 + (y - goal[1]) ** 2 + (z - goal[2]) ** 2
