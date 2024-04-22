import wandb


class CentralLogger:
    """
    Only log data to wandb at the end of each episode.
    Idempotent.
    Usage: log data using `log` method, then call `end_episode` at the end of each episode.
    """

    def __init__(self):
        self.data = {}

    def log(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def end_episode(self):
        if self.data:  # 로깅할 데이터가 있는지 확인
            wandb.log(self.data)
            self.data = {}  # 로그 후 데이터 초기화
