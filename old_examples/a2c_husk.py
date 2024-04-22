import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from craftground.wrappers.vision import VisionWrapper
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    VecVideoRecorder,
    DummyVecEnv,
)
from wandb.integration.sb3 import WandbCallback

from avoid_damage import AvoidDamageWrapper


def main():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="escape-husk-a2c-biocular",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )
    env = craftground.make(
        verbose=False,
        port=8021,
        initialInventoryCommands=[],
        initialPosition=None,  # nullable
        initialMobsCommands=[
            "minecraft:husk ~ ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            "minecraft:husk ~ ~ ~-5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            "minecraft:husk ~5 ~ ~5 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            "minecraft:husk ~10 ~ ~10 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            "minecraft:husk ~10 ~ ~-10 {HandItems:[{Count:1,id:iron_shovel},{}]}",
            # player looks at south (positive Z) when spawn
        ],
        imageSizeX=114,
        imageSizeY=64,
        visibleSizeX=114,
        visibleSizeY=64,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=True,
        alwaysNight=False,
        initialWeather="clear",  # nullable
        isHardCore=False,
        isWorldFlat=True,  # superflat world
        obs_keys=["sound_subtitles"],
        initialExtraCommands=[],
        isHudHidden=False,
        render_action=True,
        render_distance=5,
        simulation_distance=5,
        is_biocular=True,
        render_alternating_eyes=True,
        eye_distance=0.3,
    )
    env = FastResetWrapper(
        TimeLimitWrapper(
            ActionWrapper(
                AvoidDamageWrapper(VisionWrapper(env, x_dim=114, y_dim=64)),
                enabled_actions=[
                    Action.FORWARD,
                    Action.BACKWARD,
                    Action.STRAFE_LEFT,
                    Action.STRAFE_RIGHT,
                    Action.TURN_LEFT,
                    Action.TURN_RIGHT,
                ],
            ),
            max_timesteps=400,
        )
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
        "CnnPolicy",
        env,
        verbose=1,
        device="mps",
        tensorboard_log=f"runs/{run.id}",
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
    main()
