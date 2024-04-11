from typing import SupportsFloat, Any, Optional

from gymnasium.core import WrapperActType, WrapperObsType, Wrapper

from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper


# Expected structure:
# MazeSuccessWrapper(
# SparseMazeWrapper(
#     MazeSelectionWrapper()
# )
# )
class SparseMazeWrapper(Wrapper):
    def __init__(
        self,
        env: MazeReachCheckAndLogWrapper,
        reward: float,
        **kwargs,
    ):
        self.env = env
        self.reward = reward
        super().__init__(self.env)

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.env.reached_goal:
            reward += self.reward
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
