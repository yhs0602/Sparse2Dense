# 13 x 13
import wandb

from cross_w2.cross_w2_env import CROSS_W2_GOALS_INSTANCES
from post_processing.make_h_movie import create_video_from_positions

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
        "jourhyang123/craftground-sb3/23riz1rk",  # transition 1
    ]
    for run_name in run_names:
        run = api.run(run_name)
        # 로그 데이터 가져오기
        data = run.history(keys=["episode/positions", "goal_idx"], pandas=False)
        # 각 에피소드별로 동영상 생성
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            print(episode_data.keys())
            goal_idx = episode_data["goal_idx"]
            goal1 = CROSS_W2_GOALS_INSTANCES[goal_idx].pos[0]
            goal2 = CROSS_W2_GOALS_INSTANCES[goal_idx].pos[1]
            create_video_from_positions(
                cross_str,
                1,
                0,
                positions,
                f"{run.id}_{episode_id}.mp4",
                goals=[(goal1[0], goal1[2]), (goal2[0], goal2[2])],
            )
            n += 1
            if n >= 3:
                break


if __name__ == "__main__":
    make_cross_w2_movie()
