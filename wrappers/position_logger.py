import gymnasium

from utils.central_logger import CentralLogger


class PositionLoggingWrapper(gymnasium.Wrapper):
    """
    A wrapper that collects the position of the agent at each step.
    It actually logs the positions at the end of each episode.
    :param env: The environment
    """

    def __init__(self, env, logger: CentralLogger, **kwargs):
        super().__init__(env)
        self.logger = logger
        self.position_log = []

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info_obs = info["full"]
        pos = (info_obs.x, info_obs.y, info_obs.z, info_obs.yaw)
        self.position_log.append(pos)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.logger.log({"episode/positions": self.position_log})
        self.position_log = []
        return ret

    def skip_step(self):
        return self.env.skip_step()
