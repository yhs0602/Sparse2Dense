from typing import Any, Optional, Tuple, Union, Iterable, List

from gymnasium.core import WrapperObsType, Wrapper

from cross_w2.cross_w2_env import Goal
from wrappers.changing_eval_wrapper import InjectedParameterProvider


# Select maze goal
# :exports: self.maze_goal
class MazeSelectionWrapper(Wrapper):
    def __init__(
        self,
        env,
        **kwargs,
    ):
        self.env = env
        self.maze_goal: Optional[Goal] = None
        self.variable_providers: List[InjectedParameterProvider] = []
        super().__init__(self.env)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Remove cake at the goal
        if self.maze_goal:
            self.remove_cake(self.maze_goal.pos)
        self.select_goal()
        # Set cake at the goal
        self.add_cake(self.maze_goal.pos)
        return obs, info

    def select_goal(self):
        for variable_provider in self.variable_providers:
            goal = variable_provider.get_injected_variables("goal")
            if goal is not None:
                self.maze_goal = goal
            else:
                print(f"No goal provided: {variable_provider}")

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
