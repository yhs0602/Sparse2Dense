import os.path
import random
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
map_path = os.path.join(nbts_path, "room_nobell.nbt")

INITIAL_POSITION = [3.0, 2.0, 3.5, -90, 0]

GOAL_RANGE = {
    "x": (7.5, 10.5),
    "z": (14.5, 18.5),
    "y": (2.0, 2.0),
}

SPAWN_RANGE_1 = {
    "x": (2.5, 5.5),
    "z": (2.5, 5.5),
    "y": (2.0, 2.0),
}

SPAWN_RANGE_2 = {
    "x": (9.5, 11.5),
    "z": (4.5, 6.5),
    "y": (2.0, 2.0),
}

SPAWN_RANGE_3 = {
    "x": (6.5, 8.5),
    "z": (1.5, 3.5),
    "y": (2.0, 2.0),
}

SPAWN_RANGE_4 = {
    "x": (1.5, 2.5),
    "z": (12.5, 13.5),
    "y": (2.0, 2.0),
}


def select_goal_spawn():
    spawn_candidates = [
        SPAWN_RANGE_1,
        SPAWN_RANGE_2,
        SPAWN_RANGE_3,
        SPAWN_RANGE_4,
    ]
    spawn_range_idx = random.randint(0, 3)
    spawn_range = spawn_candidates[spawn_range_idx]
    spawn_x = random.uniform(spawn_range["x"][0], spawn_range["x"][1])
    spawn_z = random.uniform(spawn_range["z"][0], spawn_range["z"][1])
    spawn_y = random.uniform(spawn_range["y"][0], spawn_range["y"][1])
    goal_x = random.uniform(GOAL_RANGE["x"][0], GOAL_RANGE["x"][1])
    goal_z = random.uniform(GOAL_RANGE["z"][0], GOAL_RANGE["z"][1])
    goal_y = GOAL_RANGE["y"][0]
    return {
        "spawn_idx": spawn_range_idx,
        "spawn": (spawn_x, spawn_y, spawn_z),
        "goal": (goal_x, goal_y, goal_z),
    }


def remove_goal_command():
    return "kill @e[type=minecraft:block_display]"


def spawn_goal_command(position: Tuple[float, float, float]) -> str:
    start_x, start_y, start_z = position
    start_x -= 0.25
    start_z -= 0.25
    return (
        f"/summon block_display {start_x} {start_y} {start_z} "
        + "{transformation:{left_rotation:[0f,0f,0f,1f],right_rotation:[0f,0f,0f,1f],"
        'translation:[0f,0f,0f],scale:[0.5f,0.5f,0.5f]},block_state:{Name:"minecraft:light_blue_glazed_terracotta"}}'
    )


def define_room_metrics():
    pass


# Spawn point
# z: 2.5 ~ 5.5
# x: 2.5 ~ 5.5
# y: 2.0

# Goal point
# x: 7.5 ~ 10.5
# z: 14.5 ~ 18.5


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
# s = sand stairs
my_room_str = [
    "ttttttttttttttttttt_",
    "taaaaaggssssbBcccctt",
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
real_room_str = [row[::-1] for row in my_room_str[::-1]]
assert len(my_room_str) == 13
# Then,
# up = +x
# right = +z
assert len(real_room_str) == 13

room_palette = {
    "t": (255, 182, 193),  # Terra cotta (pinkish)
    "a": (255, 127, 80),  # Acacia wood (coral)
    "g": (169, 169, 169),  # Gray wool (dark gray)
    "v": (255, 215, 0),  # Bell (gold)
    "b": (0, 0, 255),  # Blue carpet (blue)
    "w": (255, 255, 255),  # White carpet (white)
    "A": (255, 127, 80),  # Acacia wood slab (same as acacia wood)
    "B": (255, 182, 193),  # Chiseled bookcase (pinkish)
    "c": (165, 42, 42),  # Chest (brown)
    "_": (34, 139, 34),  # Empty (green)
    "C": (220, 20, 60),  # Cherry wood (crimson)
    "p": (128, 0, 0),  # Pot (maroon)
    "d": (255, 255, 0),  # Doss (bed) (yellow)
    "O": (139, 69, 19),  # Oak chair (saddle brown)
    "K": (255, 20, 147),  # Cake (deep pink)
    "P": (192, 192, 192),  # Piston (silver)
    "i": (211, 211, 211),  # Iron trapdoor (light gray)
    "s": (210, 180, 140),  # Sand stairs (tan)
}


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
                "place template minecraft:room_nobell 0 0 0",
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
