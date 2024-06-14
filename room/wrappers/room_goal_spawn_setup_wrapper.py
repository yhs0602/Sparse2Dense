from typing import Any, Optional, Tuple, Callable, Union, Dict

from gymnasium.core import WrapperObsType, Wrapper


class RoomGoalSelectionWrapper(Wrapper):
    def __init__(
        self,
        env,
        goal_selector: Callable[[], Dict[str, Union[int, Tuple[float, float, float]]]],
        goal_set_command_provider: Callable[[Tuple[float, float, float]], str],
        goal_remove_command_provider: Callable[[], str],
        **kwargs,
    ):
        self.env = env
        self.setup_sampler = goal_selector
        self.goal_set_command_provider = goal_set_command_provider
        self.goal_remove_command_provider = goal_remove_command_provider
        self.room_settings = self.setup_sampler()
        super().__init__(self.env)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        # Remove the indicator at the goal
        self.remove_goal_indicator()
        self.room_settings = self.setup_sampler()
        # Set the indicator at the goal
        self.add_goal_indicator(self.room_settings["goal"])
        return obs, info

    def remove_goal_indicator(
        self,
    ):
        command = self.goal_remove_command_provider()
        self.get_wrapper_attr("add_commands")([command])

    def add_goal_indicator(
        self,
        goal: Tuple[float, float, float],
    ):
        command = self.goal_set_command_provider(goal)
        self.get_wrapper_attr("add_commands")([command])
