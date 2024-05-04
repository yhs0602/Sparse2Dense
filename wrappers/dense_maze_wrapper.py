from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper


# Goal 지점으로부터 일정 거리까지 (거리 5 이내) dense reward를 주고, 에피소드 종료
#
# - Reward 계산: Potential Based
#     - Taxicab distance로 계산한 거리 이용
#     - 스텝 당 리워드: 멀어졌으면 -0.01, 가까워졌으면 0.01
class DenseMazeWrapper(Wrapper):
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
        self.previous_distance = radius
        self.goal = self.get_wrapper_attr("maze_goal")
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        x = info_obs.x
        y = info_obs.y
        z = info_obs.z
        # Dense goal check
        if self.within_range(x, y, z):  # Only when the agent is in the reward range
            new_distance = self.taxicab_distance(x, y, z)
            if new_distance > self.previous_distance:
                reward -= self.reward
            elif new_distance < self.previous_distance:
                reward += self.reward
            self.previous_distance = new_distance

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def within_range(self, x, y, z):
        if len(self.goal) == 3:
            goals = [self.goal]
        else:
            goals = self.goal
        return any(self.within_range_single(goal, x, y, z) for goal in goals)

    def within_range_single(self, goal, x, y, z) -> bool:
        return (
            goal[0] - self.radius <= x <= goal[0] + self.radius
            and goal[1] - self.radius <= y <= goal[1] + self.radius
            and goal[2] - self.radius <= z <= goal[2] + self.radius
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.previous_distance = self.radius
        return obs, info

    def taxicab_distance(self, x: float, y: float, z: float) -> float:
        if len(self.goal) == 3:
            goals = [self.goal]
        else:
            goals = self.goal
        return min(self.taxicab_distance_single(goal, x, y, z) for goal in goals)

    def taxicab_distance_single(self, goal, x: float, y: float, z: float) -> float:
        return abs(x - goal[0]) + abs(y - goal[1]) + abs(z - goal[2])
