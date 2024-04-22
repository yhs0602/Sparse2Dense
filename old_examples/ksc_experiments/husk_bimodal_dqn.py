from craftground.wrappers.action import ActionWrapper, Action
from craftground.wrappers.bimodal import BimodalWrapper
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.time_limit import TimeLimitWrapper
from stable_baselines3 import DQN

from old_examples.avoid_damage import AvoidDamageWrapper
from old_examples.ksc_experiments.avoid_husk_wrapper import AvoidHuskWrapper
from old_examples.ksc_experiments.ksc_env import ContinuousHuskEnvironment
from old_examples.ksc_experiments.runner import run_ksc_experiment

if __name__ == "__main__":
    env, sound_list = ContinuousHuskEnvironment().make(
        hud=False,
        verbose=False,
        port=8001,
        render_action=True,
        size_x=114,
        size_y=64,
        render_distance=2,
        simulation_distance=5,
        strong=True,
        min_distance=15,
        max_distance=30,
        num_husks=10,
        continuous=True,
    )
    env = FastResetWrapper(
        TimeLimitWrapper(
            ActionWrapper(
                AvoidDamageWrapper(
                    AvoidHuskWrapper(
                        BimodalWrapper(env, x_dim=114, y_dim=64, sound_list=sound_list),
                        danger_reward=-0.1,
                    ),
                    damage_reward=-0.1,
                    alive_reward=0.5,
                    death_reward=-1,
                ),
                enabled_actions=[
                    Action.NO_OP,
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

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        device="mps",
        # tensorboard_log=f"runs/{run.id}",
    )

    run_ksc_experiment(
        group="ksc-journal-husk-bimodal-dqn",
        env=env,
        model=model,
    )
