import gymnasium

from utils.central_logger import CentralLogger


class LogFlushWrapper(gymnasium.Wrapper):
    def __init__(self, env, logger: CentralLogger):
        super().__init__(env)
        self.logger = logger

    def reset(self, **kwargs):
        retv = self.env.reset(**kwargs)
        self.logger.end_episode()
        return retv
