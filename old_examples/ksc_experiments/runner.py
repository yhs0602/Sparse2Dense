import gymnasium as gym
import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    DummyVecEnv,
)
from wandb.integration.sb3 import WandbCallback


def run_ksc_experiment(
    group: str,
    env: gym.Wrapper,
    model: BaseAlgorithm,
):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group=group,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )

    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 4000 == 0,
        video_length=400,
    )

    model.learn(
        total_timesteps=400000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    model.save("a2c_stack_craftground")
    run.finish()


# if __name__ == "__main__":
#     main()
