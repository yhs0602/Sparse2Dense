import wandb
from craftground import craftground
from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.sound import SoundWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from wrappers.fishing_environment import FishAnythingWrapper


def fishing():
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="craftground-sb3",
        entity="jourhyang123",
        # track hyperparameters and run metadata
        group="fish-dqn-sound",
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional    save_code=True,  # optional
    )
    size_x = 114
    size_y = 64
    env, sound_list = craftground.make(
        port=8011,
        initialInventoryCommands=[
            "fishing_rod{Enchantments:[{id:lure,lvl:3},{id:mending,lvl:1},{id:unbreaking,lvl:3}]} 1"
        ],
        initialPosition=None,  # nullable
        initialMobsCommands=[],
        imageSizeX=size_x,
        imageSizeY=size_y,
        visibleSizeX=size_x,
        visibleSizeY=size_y,
        seed=12345,  # nullable
        allowMobSpawn=False,
        alwaysDay=True,
        alwaysNight=False,
        initialWeather="rain",  # nullable
        isHardCore=False,
        isWorldFlat=False,  # superflat world
        obs_keys=["sound_subtitles"],
        miscStatKeys=["fish_caught"],
        initialExtraCommands=["tp @p -25 62 -277 127.2 -6.8"],  # x y z yaw pitch
        isHudHidden=False,
        render_action=True,
        render_distance=2,
        simulation_distance=5,
    ), [
        "subtitles.entity.experience_orb.pickup",
        "subtitles.entity.fishing_bobber.retrieve",
        "subtitles.entity.fishing_bobber.splash",
        "subtitles.entity.fishing_bobber.throw",
        "subtitles.entity.item.pickup",
    ]
    env = FastResetWrapper(
        ActionWrapper(
            FishAnythingWrapper(
                SoundWrapper(
                    env,
                    sound_list=sound_list,
                    coord_dim=2,
                ),
                reward=1,
            ),
            enabled_actions=[
                Action.NO_OP,
                Action.USE,
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
    model = DQN(
        "MlpPolicy", env, verbose=1, device="mps", tensorboard_log=f"runs/{run.id}"
    )

    model.learn(
        total_timesteps=400000,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    # model.save("dqn_sound_husk")
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
    fishing()
