from typing import Any, Optional, Tuple, Callable

from gymnasium.core import WrapperObsType, Wrapper


# Select maze goal
# :exports: self.maze_goal
class MazeSelectionWrapper(Wrapper):
    def __init__(
        self,
        env,
        goal_selector: Callable[[], Tuple[float, float, float]],
        **kwargs,
    ):
        self.env = env
        self.goal_selector = goal_selector
        self.maze_goal = self.goal_selector()
        super().__init__(self.env)

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
                f"setblock {self.maze_goal[0]} {self.maze_goal[1]} {self.maze_goal[2]} minecraft:air replace"
            ]
        )
        self.maze_goal = self.goal_selector()
        # Set cake at the goal
        self.get_wrapper_attr("add_commands")(
            [
                f"setblock {self.maze_goal[0]} {self.maze_goal[1]} {self.maze_goal[2]} minecraft:cake replace"
            ]
        )
        return obs, info
