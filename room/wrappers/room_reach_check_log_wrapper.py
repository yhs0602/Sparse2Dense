from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from utils.central_logger import CentralLogger
from wrappers.reached_goal_provider import ReachedGoalProvider

COOLDOWN = 5


class RoomReachCheckAndLogWrapper(ReachedGoalProvider, Wrapper):
    def __init__(
        self,
        env: RoomGoalSelectionWrapper,
        radius: float,
        central_logger: CentralLogger,
        cooldown: int = COOLDOWN,
        **kwargs,
    ):
        self.env = env
        self.radius = radius
        self.config_cooldown = cooldown
        self.cooldown = cooldown
        self._reached_goal = False
        self.success_counts_by_start_idx = {}
        self.time_took = 0
        self.logger = central_logger
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        x = info_obs.x
        y = info_obs.y
        z = info_obs.z
        self.cooldown -= 1
        self.time_took += 1
        self._reached_goal = False

        goal = self.env.room_settings["goal"]
        start_idx = self.env.room_settings["spawn_idx"]
        if self.cooldown <= 0:
            # square goal check
            if self.reached_goal_checker(goal, x, y, z):
                self._reached_goal = True
                print(f"Goal Reached in {self.time_took} steps")
                self.success_counts_by_start_idx[start_idx] = (
                    self.success_counts_by_start_idx.get(goal, 0) + 1
                )
                goal_str = str(goal)
                self.logger.log(
                    {
                        f"{goal_str}/success_count": self.success_counts_by_start_idx[
                            goal
                        ],
                        f"{goal_str}/time_took": self.time_took,
                        f"goal": str(goal),
                    }
                )
                self.cooldown = self.config_cooldown
                self.time_took = 0
                terminated = True
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )  # , done: deprecated

    def reached_goal_checker(self, goal, x, y, z):
        if len(goal) == 3:
            goal = [goal]
        return any(self.check_goal(g, x, y, z) for g in goal)

    def check_goal(self, goal, x, y, z):
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
        self.cooldown = self.config_cooldown
        self.time_took = 0
        self._reached_goal = False
        return obs, info

    @property
    def reached_goal(self):
        return self._reached_goal
