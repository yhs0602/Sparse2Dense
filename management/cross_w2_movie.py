# 13 x 13
import wandb

from management.make_h_movie import create_video_from_positions

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
        "jourhyang123/craftground-sb3/djuhyq2w", # dense 0
        "jourhyang123/craftground-sb3/dhpqzeij", # transition 0
        "jourhyang123/craftground-sb3/ngk59t0e", # sparse 0

    ]
    for run_name in run_names:
        run = api.run(run_name)
        # 로그 데이터 가져오기
        data = run.history(keys=["episode/positions"], pandas=False)
        # 각 에피소드별로 동영상 생성
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            create_video_from_positions(cross_str, 0, 1, positions, f"{run.id}_{episode_id}.mp4")
            n += 1
            if n >= 3:
                break


if __name__ == "__main__":
    make_cross_w2_movie()
