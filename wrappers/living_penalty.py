from typing import SupportsFloat, Any, Optional

import wandb
from gymnasium.core import WrapperActType, WrapperObsType, Wrapper


class LivingPenaltyWrapper(Wrapper):
    def __init__(
        self,
        env,
        penalty_abs: SupportsFloat,
        **kwargs,
    ):
        self.env = env
        self.penalty = penalty_abs
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward) - float(self.penalty)

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
