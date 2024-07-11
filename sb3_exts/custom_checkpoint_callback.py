import os

from stable_baselines3.common.callbacks import BaseCallback


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, steps, save_path, verbose=0):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.steps = steps
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps in self.steps:
            checkpoint_path = f"{self.save_path}/model_{self.num_timesteps}_steps"
            os.makedirs(checkpoint_path, exist_ok=True)
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(
                    f"Saving model checkpoint to {checkpoint_path} at step {self.num_timesteps}"
                )
        return True
