import gymnasium

from utils.central_logger import CentralLogger


class LogFlushWrapper(gymnasium.Wrapper):
    def __init__(self, env, logger: CentralLogger, is_eval: bool = False):
        super().__init__(env)
        self.logger = logger
        self.is_eval: bool = is_eval

    def reset(self, **kwargs):
        retv = self.env.reset(**kwargs)
        self.logger.end_episode(is_eval=self.is_eval)
        return retv
