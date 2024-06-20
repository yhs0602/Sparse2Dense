# DEPRECATED
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
    api = wandb.Api(timeout=120)
    run_names = [
        "jourhyang123/craftground-sb3/ut8b91hh",  # Sparse 0
        "jourhyang123/craftground-sb3/0ps4e1nn",  # Dense 0
        "jourhyang123/craftground-sb3/jhhmgo0f"  # 300만 0
        "jourhyang123/craftground-sb3/ktxlocs3",  # 200만 0
        "jourhyang123/craftground-sb3/u0bwff7b",  # Sparse 1
        "jourhyang123/craftground-sb3/0itxorpm",  # Dense 1
        "jourhyang123/craftground-sb3/nk54yfnh",  # 300만 1
        "jourhyang123/craftground-sb3/edsz2myn",  # 200만 1
        "jourhyang123/craftground-sb3/3r4ctsay",  # Sparse 2
        "jourhyang123/craftground-sb3/8ilqwrp5",  # Dense 2
        "jourhyang123/craftground-sb3/wocxlma6",  # 300만 2
        "jourhyang123/craftground-sb3/yxbvuc6o",  # 200만 2
    ]
    for run_name in run_names:
        # 특정 프로젝트와 run ID 지정
        # run_name = "jourhyang123/craftground-sb3/nlc1z8m1"
        run = api.run(run_name)

        # 로그 데이터 가져오기
        data = run.history(keys=["episode/positions"], pandas=False)
        # 각 에피소드별로 동영상 생성
        n = 0
        for episode_id, episode_data in enumerate(data[::-1]):
            positions = episode_data["episode/positions"]
            create_video_from_positions(cross_str, 0, 1, positions, episode_id)
            n += 1
            if n >= 3:
                break


if __name__ == "__main__":
    make_cross_w2_movie()
