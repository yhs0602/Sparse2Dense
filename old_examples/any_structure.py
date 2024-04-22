import os.path

import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback


def structure_any():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="structure-a2c-vision",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )
    size_x = 114
    size_y = 64
    base_env, sound_list = (
        craftground.make(
            port=8001,
            initialInventoryCommands=[],
            verbose=True,
            initialPosition=[5, 5, 5],  # nullable
            initialMobsCommands=[],
            imageSizeX=size_x,
            imageSizeY=size_y,
            visibleSizeX=size_x,
            visibleSizeY=size_y,
            seed=12345,  # nullable
            allowMobSpawn=False,
            alwaysDay=True,
            alwaysNight=False,
            initialWeather="clear",  # nullable
            isHardCore=False,
            isWorldFlat=True,  # superflat world
            obs_keys=[],  # No sound subtitles
            miscStatKeys=[],  # No stats
            initialExtraCommands=[
                "place template minecraft:portable_maze 0 0 0",
                "tp @p 5 5 5 -90 0",
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=True,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                os.path.abspath("../h_maze/nbt/portable_maze.nbt"),
            ],
        ),
        [],
    )
    env = FastResetWrapper(
        ActionWrapper(
            VisionWrapper(
                base_env,
                x_dim=size_x,
                y_dim=size_y,
            ),
            enabled_actions=[
                Action.FORWARD,
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
            ],
        ),
    )
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecVideoRecorder(
        env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % 4000 == 0,
        video_length=400,
    )
    model = A2C(
        "CnnPolicy", env, verbose=1, device="mps", tensorboard_log=f"runs/{run.id}"
    )

    model.learn(
        total_timesteps=1000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    # model.save("dqn_sound_husk")
    run.finish()
    base_env.terminate()

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     # vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()


if __name__ == "__main__":
    structure_any()
