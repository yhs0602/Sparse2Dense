from typing import Any, Optional, Dict

import gymnasium
from gymnasium import Env
from gymnasium.core import ObsType, ActType, WrapperObsType


class ReachCheckAndLogWrapper(gymnasium.Wrapper):
    def __init__(self, env: Env[ObsType, ActType], radius: float, **kwargs):
        super().__init__(env)
        self.reached_goal = False
        self.radius = radius

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> tuple[WrapperObsType, Dict[str, Any]]:
        self.reached_goal = False
        return self.env.reset(seed=seed, options=options)
