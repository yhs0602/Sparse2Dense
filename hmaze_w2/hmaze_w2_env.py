import os.path
from typing import Tuple

from craftground import craftground
from craftground.craftground import CraftGroundEnvironment
from craftground.craftground.screen_encoding_modes import ScreenEncodingMode

from utils.check_vglrun import check_vglrun

# ./nbt/hmaze_w2.nbt
current_folder_path = os.path.dirname(os.path.abspath(__file__))
nbts_path = os.path.join(current_folder_path, "nbt")
map_path = os.path.join(nbts_path, "hmaze_w2.nbt")

INITIAL_POSITION = [3, 1, 2.0, -90, 0]

HMAZE_W2_GOALS = [
    ((16, 1, 1), (16, 1, 2)),  # 앞쪽
    ((16, 1, 10), (13, 1, 11)),  # 앞오른쪽
    ((3, 1, 10), (7, 1, 11)),  # 뒤오른쪽
]

# 21 x 16 미로
# 17,0 ~ 17, 12
# 2, 12
# 0,0 기준으로 그리기
HMAZE_W2_STR = [
    "x" * 13,
    "x" * 13,
    "xooxxxxxxxoox",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oiioooooooiio",  # i: 휘장
    "oxxixxxxxixxo",
    "oxxixxxxxixxo",
    "oiioooooooiio",  # i: 휘장
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "oxxoxxxxxoxxo",
    "xooxxxxxxxoox",
]
assert len(HMAZE_W2_STR) == 18


def make_hmaze_w2_env(
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
                "place template minecraft:hmaze_w2 0 0 0",
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
        ),
        [],
    )
