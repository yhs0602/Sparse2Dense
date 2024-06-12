import os.path
from typing import Tuple

from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode

from utils.check_vglrun import check_vglrun

# place template room 0 -31 0
# player y: -29.0
#
# kill @e[type=minecraft:block_display]
# /summon block_display 9.25 -29 15.25 {transformation:{left_rotation:[0f,0f,0f,1f],right_rotation:[0f,0f,0f,1f],translation:[0f,0f,0f],scale:[0.5f,0.5f,0.5f]},block_state:{Name:"minecraft:light_blue_glazed_terracotta"}}

# ./nbt/room.nbt
current_folder_path = os.path.dirname(os.path.abspath(__file__))
nbts_path = os.path.join(current_folder_path, "nbt")
map_path = os.path.join(nbts_path, "room.nbt")

INITIAL_POSITION = [3.0, 2.0, 3.5, -90, 0]

# Spawn point
# z: 2.5 ~ 5.5
# x: 2.5 ~ 5.5
# y: 2.0

# Goal point
# x: 7.5 ~ 10.5
# z: 14.5 ~ 18.5


def make_room_env(
    port: int,
    size_x: int,
    size_y: int,
    verbose: bool = False,
    verbose_python: bool = False,
    verbose_gradle: bool = False,
    verbose_jvm: bool = False,
) -> Tuple[CraftGroundEnvironment, list[str]]:
    return (
        craftground.make(
            port=port,
            initialInventoryCommands=[],
            verbose=verbose,
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
                "time set noon",
                "place template minecraft:room 0 0 0",
                f"tp @p {INITIAL_POSITION[0]} {INITIAL_POSITION[1]} {INITIAL_POSITION[2]} {INITIAL_POSITION[3]} {INITIAL_POSITION[4]}",
                # "effect give @p minecraft:speed infinite 1 true",  # speed effect, particle hidden
            ],  # x y z yaw pitch
            isHudHidden=True,
            render_action=False,
            render_distance=5,
            simulation_distance=5,
            structure_paths=[
                map_path,
            ],
            no_pov_effect=True,
            screen_encoding_mode=ScreenEncodingMode.RAW,
            use_vglrun=check_vglrun(),
            verbose_python=verbose_python,
            verbose_gradle=verbose_gradle,
            verbose_jvm=verbose_jvm,
        ),
        [],
    )
