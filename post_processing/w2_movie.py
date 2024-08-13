# 13 x 13
from typing import Union, Tuple, Iterable

import wandb

from post_processing.movie import create_video_from_positions
from wandb_envs import WANDB_PROJECT, WANDB_ENTITY


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

# right = +x
# up = +z
cross_str = [
    "xxxxxxooxxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xoooooxxooooox",
    "oxxxxxxxxxxxxo",
    "oxxxxxxxxxxxxo",
    "xoooooxxooooox",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxoxxoxxxxx",
    "xxxxxxooxxxxxx",
]

assert len(cross_str) == 14


# start point: 2 32 6 > 0 0 0


def make_cross_w2_movie():
    # W&B API 초기화
    api = wandb.Api(timeout=60)
    # 특정 프로젝트와 run ID 지정
    run_names = [
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/ut8b91hh",  # Sparse 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/0ps4e1nn",  # Dense 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/jhhmgo0f"  # 300만 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/ktxlocs3",  # 200만 0
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/u0bwff7b",  # Sparse 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/0itxorpm",  # Dense 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/nk54yfnh",  # 300만 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/edsz2myn",  # 200만 1
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/3r4ctsay",  # Sparse 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/8ilqwrp5",  # Dense 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/wocxlma6",  # 300만 2
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/yxbvuc6o",  # 200만 2
    ]
    for run_name in run_names:
        run = api.run(run_name)
        # 로그 데이터 가져오기
        data = run.history(
            keys=[
                "eval_episode/positions",
                # "eval_goal_idx",
                "eval_episode",
                # "eval_reached_goal",
                "eval_episode/goal",
            ],
            pandas=False,
        )
        # 각 에피소드별로 동영상 생성
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["eval_episode/positions"]
            # print(episode_data.keys())
            # goal_idx = episode_data["eval_goal_idx"]
            # reached_goal = episode_data["eval_reached_goal"]
            # goal1 = CROSS_W2_GOALS_INSTANCES[goal_idx].pos[0]
            # goal2 = CROSS_W2_GOALS_INSTANCES[goal_idx].pos[1]
            reached_goal = True  # Unkown
            goal1 = episode_data["eval_episode/goal"][0]
            goal2 = episode_data["eval_episode/goal"][1]
            create_video_from_positions(
                cross_str,
                1,
                0,
                positions,
                f"{run.group}_{run.id}_{episode_id}.mp4",
                episode=episode_data["eval_episode"],
                goals=[(goal1[0], goal1[2]), (goal2[0], goal2[2])],
                reached_goal=reached_goal,
            )
            n += 1
            if n >= 3:
                break
        else:
            print("No data")


if __name__ == "__main__":
    make_cross_w2_movie()
