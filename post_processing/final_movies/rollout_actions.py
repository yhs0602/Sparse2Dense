import json
import os
from typing import Union, Dict, Tuple

import gymnasium
from craftground.wrappers.fast_reset import FastResetWrapper
from craftground.wrappers.vision import VisionWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from cross_w2.cross_w2_env import make_cross_w2_env
from room.room_env import make_room_env, spawn_goal_command, remove_goal_command
from room.wrappers.room_goal_spawn_setup_wrapper import RoomGoalSelectionWrapper
from room.wrappers.room_reach_check_log_wrapper import RoomReachCheckAndLogWrapper
from utils.central_logger import CentralLogger
from wrappers.maze_reach_wrapper import MazeReachCheckAndLogWrapper
from wrappers.maze_selection_wrapper import MazeSelectionWrapper
from wrappers.turn_90_wrapper import Turn90Wrapper

current_folder = os.path.dirname(os.path.abspath(__file__))
ego_media_folder = os.path.join(current_folder, "ego_media")
os.makedirs(ego_media_folder, exist_ok=True)
room_media_folder = os.path.join(current_folder, "room_media")


def fixed_select_room_goal_spawn(
    spawn, goals
) -> Dict[str, Union[int, Tuple[float, float, float]]]:
    spawn_x = spawn[0]
    spawn_y = spawn[1]
    spawn_z = spawn[2]
    goal_x = goals[0][0]
    goal_y = goals[0][1]
    goal_z = goals[0][2]
    return {
        "spawn_idx": 0,
        "spawn": (spawn_x, spawn_y, spawn_z),
        "goal": (goal_x, goal_y, goal_z),
    }


def wrap_room_env(
    env, size_x, size_y, central_logger: CentralLogger, spawn, goal
) -> gymnasium.Env:
    return FastResetWrapper(
        # Truncate the episode if it takes too long
        TimeLimit(
            # Checks, Logs, Terminates
            RoomReachCheckAndLogWrapper(
                # Select goal when reset
                RoomGoalSelectionWrapper(
                    Turn90Wrapper(
                        VisionWrapper(
                            env,
                            x_dim=size_x,
                            y_dim=size_y,
                        ),
                    ),
                    logger=central_logger,
                    goal_selector=lambda: fixed_select_room_goal_spawn(spawn, goal),
                    goal_set_command_provider=spawn_goal_command,
                    goal_remove_command_provider=remove_goal_command,
                ),
                radius=2,
                central_logger=central_logger,
                cooldown=2,
            ),
            max_episode_steps=20000,
        ),
        logger=central_logger,
    )


def wrap_cross_env(
    env, size_x, size_y, central_logger: CentralLogger, goals
) -> gymnasium.Env:
    tuple_goals = tuple((goal[0], goal[1], goal[2]) for goal in goals)
    return FastResetWrapper(
        # Truncate the episode if it takes too long
        TimeLimit(
            # Checks, Logs, Terminates
            MazeReachCheckAndLogWrapper(
                # Select goal when reset
                MazeSelectionWrapper(
                    Turn90Wrapper(
                        VisionWrapper(
                            env,
                            x_dim=size_x,
                            y_dim=size_y,
                        ),
                    ),
                    logger=central_logger,
                    goal_selector=lambda: tuple_goals,
                    goal_set_command_provider=spawn_goal_command,
                    goal_remove_command_provider=remove_goal_command,
                ),
                radius=2,
                central_logger=central_logger,
                cooldown=2,
            ),
            max_episode_steps=20000,
        ),
        logger=central_logger,
    )


def main(port1: int):
    size_x = 114
    size_y = 64
    # for each json in room_media_folder
    for idx, json_file in enumerate(os.listdir(room_media_folder)):
        port = port1 + idx
        if not json_file.endswith(".json"):
            print(f"Skipping {json_file}")
            continue

        # load the json
        content = json.load(open(os.path.join(room_media_folder, json_file)))
        # get the positions
        positions = content["positions"]
        # get the goals
        goals = content["goal"]
        actions = content["actions"]

        # Setup environment
        try:
            central_logger = CentralLogger()
            if "cross" in json_file:
                base_env, _ = make_cross_w2_env(
                    port,
                    size_x,
                    size_y,
                    verbose=False,
                    verbose_python=False,
                    verbose_gradle=False,
                    verbose_jvm=False,
                )
                base_env = wrap_cross_env(
                    base_env, size_x, size_y, central_logger, goals
                )
            elif "room" in json_file:
                base_env, _ = make_room_env(
                    port,
                    size_x,
                    size_y,
                    extended=False,
                    verbose=False,
                    verbose_python=False,
                    verbose_gradle=False,
                    verbose_jvm=False,
                )
                base_env = wrap_room_env(
                    base_env, size_x, size_y, central_logger, positions[0], goals
                )
            else:
                print(f"Unknown group: {json_file}")
                continue

            env = Monitor(base_env)
            env = DummyVecEnv([lambda: base_env])
            env = VecVideoRecorder(
                env,
                os.path.join(ego_media_folder, json_file.replace(".json", "")),
                record_video_trigger=lambda x: x % 20000 == 0,
                video_length=len(actions) + 1,
            )

            obs = env.reset()
            obs = env.reset()
            obs = env.reset()
            for i in range(len(actions)):
                action = actions[i]
                obs, reward, done, info = env.step([action])
                if done:
                    print(f"Done at {i}!!!!!!!")
        finally:
            env.close()


if __name__ == "__main__":
    main(8000)
