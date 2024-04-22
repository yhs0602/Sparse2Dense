# 13 x 13
import wandb

from management.make_movie import create_video_from_positions

cross_str = [
    "xxxxxoooxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "ooooooxoooooo",
    "oxxxxxxxxxxxo",
    "ooooooxoooooo",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoxoxxxxx",
    "xxxxxoooxxxxx",
]

assert len(cross_str) == 13


# start point: 2 32 6 > 0 0 0


def make_cross_movie():
    # W&B API 초기화
    api = wandb.Api()

    # 특정 프로젝트와 run ID 지정
    project_name = "craftground-sb3"
    run_id = "zypbuddd"
    run = api.run(f"{project_name}/{run_id}")

    # 로그 데이터 가져오기
    data = run.history(keys=["episode/positions"], pandas=False)
    # 각 에피소드별로 동영상 생성
    n = 0
    for episode_id, episode_data in enumerate(data):
        positions = episode_data["episode/positions"]
        create_video_from_positions(cross_str, 1, 0, positions, episode_id)
        n += 1
        if n >= 3:
            break
