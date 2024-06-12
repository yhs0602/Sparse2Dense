# 13 x 13

import wandb

from post_processing.movie import create_video_from_positions
from room.room_env import real_room_str



def make_room_movie():
    # W&B API 초기화
    api = wandb.Api(timeout=60)

    # 특정 프로젝트와 run ID 지정
    run_names = [
        "jourhyang123/craftground-sb3/r19yjutb",  # transition 1
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
            reached_goal = True  # Unkown
            goal1 = episode_data["eval_episode/goal"][0]
            goal2 = episode_data["eval_episode/goal"][1]
            create_video_from_positions(
                real_room_str,
                1,
                0,
                positions,
                f"{run.id}_{episode_id}.mp4",
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
