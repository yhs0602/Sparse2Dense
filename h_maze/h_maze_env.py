import os.path
from typing import Tuple

from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode

from utils.check_vglrun import check_vglrun

# ./nbt/hmaze1_colored.nbt
current_folder_path = os.path.dirname(os.path.abspath(__file__))
nbts_path = os.path.join(current_folder_path, "nbt")
map_path = os.path.join(nbts_path, "hmaze1_colored.nbt")

H_MAZE_GOALS = [
    (21, 1, 1),  # 앞쪽
    (21, 1, 14),  # 앞 오른쪽
    (3, 1, 14),  # 뒤 오른쪽
]


def make_h_maze_env(
    port: int, size_x: int, size_y: int
) -> Tuple[CraftGroundEnvironment, list[str]]:
    return (
        craftground.make(
            port=port,
            initialInventoryCommands=[],
            verbose=False,
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
                "place template minecraft:hmaze1_colored 0 0 0",
                "tp @p 3 1 1 -90 0",
                "effect give @p minecraft:speed infinite 2 true",  # speed effect, particle hidden
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
        ),
        [],
    )
