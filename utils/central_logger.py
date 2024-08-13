from typing import Dict, Any

import wandb


class CentralLogger:
    """
    Only log data to wandb at the end of each episode.
    Not Idempotent.
    Usage: log data using `log` method, then call `end_episode` at the end of each episode.
    """

    def __init__(self):
        self.data = {}

    def log(self, log_dict: Dict[str, Any]):
        """
        Log data to be logged at the end of the episode.
        :param log_dict: A dictionary containing the data to be logged.
        """
        self.data.update(log_dict)

    def end_episode(self, is_eval: bool = False):
        if self.data:  # Make sure you have data to log
            if is_eval:
                prepended_dict = {
                    f"eval_{key}": value for key, value in self.data.items()
                }
                self.data = prepended_dict
            wandb.log(self.data)
            self.data = {}  # Wiping data after logging
