from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, EventCallback


class EpisodeStartCallback(EventCallback):
    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EpisodeStartCallback, self).__init__(callback, verbose)
        self.is_in_episode = False

    def _on_step(self) -> bool:
        if not self.is_in_episode:
            self.is_in_episode = True
            return self._on_event()
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        truncated = False
        if infos:
            truncated = infos[0]["TimeLimit.truncated"]
        done_ = self.locals.get("done_")
        if (dones and dones[0]) or truncated:
            self.is_in_episode = False
        # ... self._on_event()
        return True
