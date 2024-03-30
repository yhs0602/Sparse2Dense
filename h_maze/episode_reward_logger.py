import wandb
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeRewardLogger, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 에피소드가 종료될 때마다 실행
        # print(self.locals.keys())
        reward = self.locals.get("rewards")
        if reward is not None:
            self.episode_rewards.append(reward)
        if self.locals.get("done_"):
            print(
                f"done={self.locals.get('done_')} dones={self.locals.get('dones')} rewards={self.locals.get('rewards')}"
            )
            total_reward = sum(self.episode_rewards)
            wandb.log({"episode_reward": total_reward})
            self.episode_rewards = []
        return True
