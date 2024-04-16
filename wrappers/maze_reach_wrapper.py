from typing import SupportsFloat, Any, Optional

import wandb
from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

from wrappers.maze_selection_wrapper import MazeSelectionWrapper


# Expected structure:
# MazeSuccessWrapper(
# SparseMazeWrapper(
#     MazeSelectionWrapper()
# )
# )

COOLDOWN = 5


class MazeReachCheckAndLogWrapper(Wrapper):
    def __init__(
        self,
        env: MazeSelectionWrapper,
        radius: float,
        **kwargs,
    ):
        self.env = env
        self.radius = radius
        self.cooldown = COOLDOWN
        self.reached_goal = False
        self.success_counts = {}
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
        self.reached_goal = False

        goal = self.env.maze_goal
        if self.cooldown <= 0:
            # square goal check
            if (
                goal[0] - self.radius <= x <= goal[0] + self.radius
                and goal[1] - self.radius <= y <= goal[1] + self.radius
                and goal[2] - self.radius <= z <= goal[2] + self.radius
            ):
                self.reached_goal = True
                print(f"Goal Reached in {self.time_took} steps")
                self.success_counts[goal] = self.success_counts.get(goal, 0) + 1
                goal_str = str(goal)
                wandb.log(
                    {
                        f"{goal_str}/success_count": self.success_counts[goal],
                        f"{goal_str}/time_took": self.time_took,
                    }
                )
                self.cooldown = COOLDOWN
                self.time_took = 0
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
        self.cooldown = COOLDOWN
        self.time_took = 0
        self.reached_goal = False
        return obs, info
