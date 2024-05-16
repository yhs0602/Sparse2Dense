import os.path
from typing import Tuple, Union, Iterable

from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode

from utils.check_vglrun import check_vglrun

# ./nbt/cross_w2.nbt
current_folder_path = os.path.dirname(os.path.abspath(__file__))
nbts_path = os.path.join(current_folder_path, "nbt")
map_path = os.path.join(nbts_path, "cross_w2.nbt")

INITIAL_POSITION = [2, 1, 6.5, -90, 0]


class Goal:
    def __init__(
        self,
        idx: int,
        pos: Union[Tuple[float, float, float], Iterable[Tuple[float, float, float]]],
        name: str,
    ):
        self.idx = idx
        self.pos = pos
        self.name = name

    def __str__(self):
        return f"Goal {self.name}({self.idx}): {self.pos}"

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


CROSS_W2_GOALS = [
    ((7, 1, 1), (8, 1, 1)),  # 왼쪽
    ((13, 1, 6), (13, 1, 7)),  # 앞쪽
    ((8, 1, 12), (7, 1, 12)),  # 오른쪽
]

CROSS_W2_GOALS_INSTANCES = [
    Goal(0, CROSS_W2_GOALS[0], "Left"),
    Goal(1, CROSS_W2_GOALS[1], "Front"),
    Goal(2, CROSS_W2_GOALS[2], "Right"),
]


def make_cross_w2_env(
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
                "place template minecraft:cross_w2 0 0 0",
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
