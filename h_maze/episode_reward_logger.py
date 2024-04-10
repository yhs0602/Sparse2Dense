import wandb
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(EpisodeLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.episode = 0
        self.episode_length = 0
        wandb.define_metric("episode")
        wandb.define_metric("episode_reward", step_metric="episode")
        wandb.define_metric("episode_length", step_metric="episode")

    def _on_step(self) -> bool:
        # 에피소드가 종료될 때마다 실행
        # print(self.locals.keys())
        reward = self.locals.get("rewards")
        if reward is not None:
            self.episode_rewards.append(reward)
        self.episode_length += 1
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        truncated = False
        if infos:
            truncated = infos[0]["TimeLimit.truncated"]
        done_ = self.locals.get("done_")
        if (dones and dones[0]) or truncated:
            self.episode += 1
            print(
                f"done={self.locals.get('done_')} dones={self.locals.get('dones')} rewards={self.locals.get('rewards')}"
            )
            total_reward = sum(self.episode_rewards)
            wandb.log(
                {
                    "episode_reward": total_reward,
                    "episode": self.episode,
                    "episode_length": self.episode_length,
                }
            )
            self.episode_rewards = []
            self.episode_length = 0
        return True
