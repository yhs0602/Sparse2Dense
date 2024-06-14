from abc import ABC, abstractmethod

from gymnasium import Wrapper


class ReachedGoalProvider(ABC, Wrapper):
    @property
    @abstractmethod
    def reached_goal(self) -> bool:
        pass
