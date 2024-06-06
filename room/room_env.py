import os.path
from typing import Tuple

from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode

from cross_w2.cross_w2_env import Goal
from utils.check_vglrun import check_vglrun

# ./nbt/room.nbt
current_folder_path = os.path.dirname(os.path.abspath(__file__))
nbts_path = os.path.join(current_folder_path, "nbt")
map_path = os.path.join(nbts_path, "room.nbt")

INITIAL_POSITION = [2, 1, 6.5, -90, 0]

# right = +x (currently -z)
# up = +z (currently -x)
# t = terra cotta
# a = acacia wood
# g = gray wool
# v = bell
# b = blue carpet
# w = white carpet
# A = acasia wood slab
# B = chiseled bookcase
# c = chest
# _ = empty
# C = cherry wood
# p = pot
# d = doss (bed)
# O = oak chair
# K = cake
# P = piston
# i = iron trapdoor
my_room_str = [
    "ttttttttttttttttttt_",
    "taaaaagvssssbBcccctt",
    "taAAAaggssssbBggggCt",
    "tgggggggbwbwbBggggCt",
    "tggggggggggggBggggCt",
    "tggggggggggggpggggCt",
    "tggggggggggggggggggt",
    "tgggggtggggggggggggt",
    "tgggggtggggggggggggt",
    "tgggggtOgKPgOgggddgt",
    "tgggggtggPPgggggddgt",
    "tgiiigtggggggggggggt",
    "tttttttttttttttttttt",
]
real_room_str = [[row[::-1] for row in my_room_str[::-1]]]
assert len(my_room_str) == 13
# Then,
# up = +x
# right = +z
assert len(real_room_str) == 13

ROOM_GOALS = [
    ((7, 1, 1), (8, 1, 1)),  # 왼쪽
]

ROOM_GOALS_INSTANCES = [
    Goal(0, ROOM_GOALS[0], "Left"),
]


def make_room_env(
    port: int,
    size_x: int,
    size_y: int,
    verbose: bool = False,
    verbose_python: bool = False,
    verbose_gradle: bool = True,
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
