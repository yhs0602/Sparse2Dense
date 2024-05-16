import abc
from abc import abstractmethod
from typing import SupportsFloat, Any, Optional, Dict, Callable

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

from utils.central_logger import CentralLogger


class InjectedParameterProvider(abc.ABC):
    @abstractmethod
    def get_injected_variables(self, name: str) -> Any:
        pass


# episode를 추적해야함
# reset횟수에 기반하여 파라미터를 바꿔가며 주입.


class InjectedParameter:
    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters


# 예시: period = 60
# ChangingEvalWrapper(
#     eval_env,
#     parameters=lambda reset_count: InjectedParameter(
#         {
#             "goal_idx": (reset_count - 1) % 3, # 0, 1, 2, 0, 1, 2, ...
#             "enabled_earlystop": (reset_count - 1) // 30 == 0, # True * 30, False * 30, ...
#             "enabled_negative_reward": (reset_count - 1) // 30 == 0, # True * 30, False * 30, ...
#         }
#     ),
#     period=60,
# )
class ChangingEvalWrapper(Wrapper, InjectedParameterProvider):
    def __init__(
        self,
        eval_env: gymnasium.Env,
        parameters: Callable[[int], InjectedParameter],
        period: int,
        central_logger: CentralLogger,
        **kwargs,
    ):
        self.env = eval_env
        self.reset_count = 0
        self.period = period
        self.parameters = parameters
        self.central_logger = central_logger
        self.injected_variables = {}
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
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
        self.reset_count += 1
        # Inject the appropriate parameters to the eval_env
        # such as the goal, enabled_earlystop, enabled_negative_reward, etc.
        parameter = self.parameters(self.reset_count)
        self.injected_variables = parameter.parameters
        # Should log to central_logger before calling reset
        self.central_logger.log(
            {
                "reset_count": self.reset_count,
            }
        )
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def get_injected_variables(self, name: str) -> Any:
        return self.injected_variables.get(name)
