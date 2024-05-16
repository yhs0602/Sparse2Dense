from typing import SupportsFloat, Any, Optional, List, Union, Tuple, Iterable

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

from cross_w2.cross_w2_env import Goal
from utils.central_logger import CentralLogger
from wrappers.changing_eval_wrapper import InjectedParameterProvider

COOLDOWN = 5


# If the wrong goal reached then give penalty and terminate the episode
class WrongGoalPenaltyWrapper(Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        total_goals: List[Goal],
        radius: float,
        central_logger: CentralLogger,
        cooldown: int = COOLDOWN,
        reward: float = -1,
        **kwargs,
    ):
        self.env = env
        self.radius = radius
        self.config_cooldown = cooldown
        self.cooldown = cooldown
        self.logger = central_logger
        self.variable_providers: List[InjectedParameterProvider] = []
        self.total_goals: List[Goal] = total_goals
        self.reward = reward
        self.correct_goal = self.get_wrapper_attr("maze_goal")
        self.enabled_earlystop = False
        self.enabled_negative_reward = False
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

        if not self.enabled_earlystop and not self.enabled_negative_reward:
            return (
                obs,
                reward,
                terminated,
                truncated,
                info,
            )

        if self.cooldown <= 0:
            # square goal check
            for wrong_goal in self.total_goals:
                if wrong_goal == self.correct_goal:
                    continue
                if self.reached_goal_checker(wrong_goal.pos, x, y, z):
                    print(f"Wrong Goal Reached: {wrong_goal}")
                    self.cooldown = self.config_cooldown
                    if self.enabled_earlystop:
                        terminated = True
                    if self.enabled_negative_reward:
                        reward += self.reward
                    break
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
        self.check_enabled()
        self.logger.log(
            {
                "enabled_earlystop": self.enabled_earlystop,
                "enabled_negative_reward": self.enabled_negative_reward,
            }
        )
        obs, info = self.env.reset(seed=seed, options=options)
        self.cooldown = self.config_cooldown
        self.correct_goal = self.get_wrapper_attr("maze_goal")
        return obs, info

    def check_enabled(self):
        for provider in self.variable_providers:
            enabled_earlystop = provider.get_injected_variables("enabled_earlystop")
            if enabled_earlystop is not None:
                self.enabled_earlystop = enabled_earlystop
            enabled_negative_reward = provider.get_injected_variables(
                "enabled_negative_reward"
            )
            if enabled_negative_reward is not None:
                self.enabled_negative_reward = enabled_negative_reward
