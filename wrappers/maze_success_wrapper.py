from typing import SupportsFloat, Any, Optional, Tuple, Callable

import wandb
from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

COOLDOWN = 5


class MazeSuccessWrapper(Wrapper):
    def __init__(
        self,
        env,
        radius: float,
        reward: float,
        goal_selector: Callable[[], Tuple[float, float, float]],
        **kwargs,
    ):
        self.env = env
        self.radius = radius
        self.reward = reward
        self.success_counts = {}
        self.goal_selector = goal_selector
        self.goal = self.goal_selector()
        self.cooldown = 0
        self.time_took = 0
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

        if self.cooldown <= 0:
            # square goal check
            if (
                self.goal[0] - self.radius <= x <= self.goal[0] + self.radius
                and self.goal[1] - self.radius <= y <= self.goal[1] + self.radius
                and self.goal[2] - self.radius <= z <= self.goal[2] + self.radius
            ):
                reward += self.reward
                print(f"Goal Reached in {self.time_took} steps")
                self.success_counts[self.goal] = (
                    self.success_counts.get(self.goal, 0) + 1
                )
                goal_str = str(self.goal)
                wandb.log(
                    {
                        f"{goal_str}/success_count": self.success_counts[self.goal],
                        f"{goal_str}/time_took": self.time_took,
                    }
                )
                terminated = True
                self.cooldown = COOLDOWN
                self.time_took = 0

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
        # Remove cake at the goal
        self.get_wrapper_attr("add_commands")(
            [
                f"setblock {self.goal[0]} {self.goal[1]} {self.goal[2]} minecraft:air replace"
            ]
        )
        self.goal = self.goal_selector()
        # Set cake at the goal
        self.get_wrapper_attr("add_commands")(
            [
                f"setblock {self.goal[0]} {self.goal[1]} {self.goal[2]} minecraft:cake replace"
            ]
        )
        self.time_took = 0
        self.cooldown = COOLDOWN
        return obs, info
