from typing import Any, Optional, Tuple, Callable, Union, Iterable

from gymnasium.core import WrapperObsType, Wrapper


# Select maze goal
# :exports: self.maze_goal
class MazeSelectionWrapper(Wrapper):
    def __init__(
        self,
        env,
        goal_selector: Callable[
            [], Union[Tuple[float, float, float], Iterable[Tuple[float, float, float]]]
        ],
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
        self.remove_cake(self.maze_goal)
        self.maze_goal = self.goal_selector()
        # Set cake at the goal
        self.add_cake(self.maze_goal)
        return obs, info

    def remove_cake(
        self,
        goal: Union[Tuple[float, float, float], Iterable[Tuple[float, float, float]]],
    ):
        if len(goal) == 3:
            goal = [goal]
        for g in goal:
            self.get_wrapper_attr("add_commands")(
                [f"setblock {g[0]} {g[1]} {g[2]} minecraft:air replace"]
            )

    def add_cake(
        self,
        goal: Union[Tuple[float, float, float], Iterable[Tuple[float, float, float]]],
    ):
        if len(goal) == 3:
            goal = [goal]
        for g in goal:
            self.get_wrapper_attr("add_commands")(
                [f"setblock {g[0]} {g[1]} {g[2]} minecraft:cake replace"]
            )
