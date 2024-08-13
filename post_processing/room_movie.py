# 13 x 13

import wandb

from post_processing.movie import create_video_from_positions
from room.room_env import real_room_str, room_palette
from wandb_envs import WANDB_PROJECT, WANDB_ENTITY


def make_room_movie():
    # W&B API 초기화
    api = wandb.Api(timeout=120)

    # 특정 프로젝트와 run ID 지정
    run_names = [
        f"{WANDB_ENTITY}/{WANDB_PROJECT}/fnsv0j1p",  # transition 2M
    ]
    for run_name in run_names:
        run = api.run(run_name)
        # 로그 데이터 가져오기
        data = run.history(
            keys=[
                "episode/positions",
                "episode/spawn",
                "episode",
                # "eval_reached_goal",
                "episode/goal",
            ],
            pandas=False,
        )
        # 각 에피소드별로 동영상 생성
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            reached_goal = True  # Unkown
            goal1 = episode_data["episode/goal"]
            print(f"Goal:{goal1}")
            create_video_from_positions(
                real_room_str,
                1,
                0,
                positions,
                f"{run.id}_{episode_id}.mp4",
                episode=episode_data["episode"],
                goals=[(int(goal1[0]), int(goal1[2]))],
                reached_goal=reached_goal,
                palette=room_palette,
            )
            n += 1
            if n >= 3:
                break
        else:
            print("No data")


if __name__ == "__main__":
    make_room_movie()
