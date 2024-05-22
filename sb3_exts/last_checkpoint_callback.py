import glob
import os

from stable_baselines3.common.callbacks import CheckpointCallback


class LastCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix="model", keep_last=2):
        super().__init__(save_freq, save_path, name_prefix)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        result = super()._on_step()
        if result:
            # Get list of checkpoint files
            checkpoint_files = glob.glob(
                os.path.join(self.save_path, f"{self.name_prefix}_*.zip")
            )
            # Sort files by creation time
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            # Remove older checkpoints, keeping only the last `keep_last` files
            for old_checkpoint in checkpoint_files[self.keep_last :]:
                os.remove(old_checkpoint)
        return result
