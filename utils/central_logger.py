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
        self.eval_rewards = []
        self.eval_time_tooks = []
        self.eval_success_counts = 0
        self.eval_count = 0

    def log(self, log_dict: Dict[str, Any]):
        """
        Log data to be logged at the end of the episode.
        :param log_dict: A dictionary containing the data to be logged.
        """
        self.data.update(log_dict)

    def end_episode(self, is_eval: bool = False):
        if self.data:  # 로깅할 데이터가 있는지 확인
            # check aggregate data
            if "goal_str" in self.data and "episode" in self.data:
                goal_str = self.data["goal_str"]
                if f"{goal_str}/success_count" in self.data:
                    episode = self.data["episode"]
                    self.data[f"{goal_str}/success_rate"] = (
                        self.data[f"{goal_str}/success_count"] / episode
                    )
            if "success_count" in self.data and "episode" in self.data:
                if "time_took" in self.data:
                    self.data["success_rate"] = (
                        self.data["success_count"] / self.data["episode"]
                    )

            if is_eval:
                skip_logging = self.data.get("skip_logging", False)
                if not skip_logging:
                    self.eval_count += 1
                    # check earlystop
                    early_stop = self.data["enabled_earlystop"]
                    enabled_negative_reward = self.data["enabled_negative_reward"]
                    early_str = "Earlystop" if early_stop else "Nostop"
                    # Parse _time_took, _reward, _success_rate
                    self.data[f"{early_str}_time_took"] = self.data["time_took"]
                    self.data[f"{early_str}_reward"] = self.data["episode/reward"]
                    self.data[f"{early_str}_success_rate"] = self.data["success_rate"]
                    self.data["enabled_earlystop"] = 1 if early_stop else 0
                    self.data["enabled_negative_reward"] = (
                        1 if enabled_negative_reward else 0
                    )

                    # Must have reset_count variable
                    reset_count = self.data["reset_count"]
                    eval_group_idx = reset_count // 60
                    self.data["group_idx"] = eval_group_idx

                    self.eval_time_tooks.append(self.data["time_took"])
                    self.eval_rewards.append(self.data["episode/reward"])
                    if self.data["reached_goal"] == 1:
                        self.eval_success_counts += 1
                    if self.eval_count == 30:
                        # Calculate eval_group_idx/{early}_mean_time_took
                        assert len(self.eval_time_tooks) == 30
                        early_mean_time_took = sum(self.eval_time_tooks) / 30
                        self.data[
                            f"{eval_group_idx}/{early_str}_mean_time_took"
                        ] = early_mean_time_took
                        self.eval_time_tooks = []
                        # Calculate eval_group_idx/{early}_mean_reward
                        assert len(self.eval_rewards) == 30
                        early_mean_reward = sum(self.eval_rewards) / 30
                        self.data[
                            f"{eval_group_idx}/{early_str}_mean_reward"
                        ] = early_mean_reward
                        self.eval_rewards = []
                        # Calculate eval_group_idx/{early}_success_rate
                        early_success_rate = self.eval_success_counts / 30
                        self.data[
                            f"{eval_group_idx}/{early_str}_success_rate"
                        ] = early_success_rate
                        self.eval_success_counts = 0
                        self.eval_count = 0

                prepended_dict = {
                    f"eval_{key}": value for key, value in self.data.items()
                }
                self.data = prepended_dict
            wandb.log(self.data)
            self.data = {}  # 로그 후 데이터 초기화
