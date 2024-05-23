import gymnasium

from utils.central_logger import CentralLogger


class PositionAndRewardLoggingWrapper(gymnasium.Wrapper):
    """
    A wrapper that collects the position of the agent at each step.
    It actually logs the positions at the end of each episode.
    :param env: The environment
    """

    def __init__(self, env, logger: CentralLogger, **kwargs):
        super().__init__(env)
        self.logger = logger
        self.position_log = []
        self.reward_log = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["obs"]
        pos = (info_obs.x, info_obs.y, info_obs.z, info_obs.yaw)
        self.position_log.append(pos)
        self.reward_log.append(reward)
        return observation, reward, terminated, truncated, info

    # CentralLogger waits for all the child wrappers to complete reset before logging the episode.
    # Initial reset does not have any content.
    # logged episode always starts from 2?
    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.logger.log(
            {
                "episode/positions": self.position_log,
                "episode/rewards": self.reward_log,
            }
        )
        self.position_log = []
        self.reward_log = []
        return ret
