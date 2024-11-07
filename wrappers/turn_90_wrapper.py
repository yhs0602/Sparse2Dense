from enum import Enum
from typing import SupportsFloat, Any, List, Optional

import gymnasium as gym
from craftground.craftground.minecraft import no_op
from gymnasium.core import WrapperActType, WrapperObsType


# Converts the int action space to a box action space
# convert int to box
class Action(Enum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


class Turn90Wrapper(gym.Wrapper):
    enabled_actions: List[Action]

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.no_op = no_op
        self.action_space = gym.spaces.Discrete(3)
        self.enabled_actions = [Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action_enum: Action = self.enabled_actions[action]
        action_arr = self.int_to_action(action_enum)
        # print(f"Final Action: {action_arr}")
        obs, reward, terminated, truncated, info = self.env.step(action_arr)

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
        return obs, info

    def int_to_action(self, input_act: Action) -> List[float]:
        act = no_op()
        # act=0: no op
        if input_act == Action.FORWARD:  # go forward
            act[0] = 1  # 0: noop 1: forward 2 : back
        elif input_act == Action.TURN_LEFT:  # Turn left
            act[4] = 12 - 6  # Camera delta yaw (0: -180, 24: 180)
        elif input_act == Action.TURN_RIGHT:  # Turn right
            act[4] = 12 + 6  # Camera delta yaw (0: -180, 24: 180)
        return act

    def skip_step(self):
        return self.env.step(action=no_op())
