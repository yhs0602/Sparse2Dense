import os

import pandas as pd
import wandb


# WandB 실행을 재개
def main(long_run_id: str):
    # get the short run id
    run_id = long_run_id.split("/")[-1]
    # Check if the cached csv exists
    if os.path.exists(f"cache/{run_id}.csv"):
        data = pd.read_csv(f"cache/{run_id}.csv")
    else:
        api = wandb.Api()
        run = api.run(long_run_id)
        data = run.history()  # 모든 로깅 데이터를 포함하는 DataFrame을 반환
        data.to_csv(f"cache/{run_id}.csv")

    # Process the data
    # select rows where eval_episode is not None
    eval_data = data[data.eval_episode.notnull()]
    # select rows where eval_episode is not 30x + 1
    eval_data = eval_data[eval_data.eval_episode.apply(lambda x: x % 30 != 1)]
    eval_data["batch"] = eval_data["eval_episode"] // 30
    grouped = eval_data.groupby(["batch", "eval_episode/goal"])
    # 평균 계산
    mean_data = grouped.agg(
        {"eval_episode/length": "mean", "eval_episode/reward": "mean"}
    )

    # 인덱스 이름을 'goal'로 변경
    mean_data = mean_data.rename_axis(["Batch", "Goal"])
    # # Group by eval_episode % 3
    # grouped = eval_data.groupby(eval_data["eval_episode/goal"])
    # # Calculate the mean of eval_episode/length, eval_episode/reward
    # mean_data = grouped.agg({"eval_episode/length": "mean", "eval_episode/reward": "mean"})
    # mean_data = mean_data.rename_axis("goal")
    print(mean_data)

    # wandb.init(project="your_project_name", id=run_id, resume="must")


if __name__ == "__main__":
    main("your_project_name/data")


# Example Output
#                                 eval_episode/length  eval_episode/reward
# Batch Goal
# 0.0   [[13, 1, 6], [13, 2, 7]]          2065.000000             0.793500
#       [[7, 1, 1], [8, 1, 1]]            1636.444444             0.836356
#       [[8, 1, 12], [7, 1, 12]]          1600.333333             0.839967
# 1.0   [[13, 1, 6], [13, 2, 7]]           771.111111             0.922889
#       [[7, 1, 1], [8, 1, 1]]            1075.666667             0.892433
#       [[8, 1, 12], [7, 1, 12]]           942.900000             0.905710
# 2.0   [[13, 1, 6], [13, 2, 7]]           373.000000             0.962700
#       [[8, 1, 12], [7, 1, 12]]          1781.000000             0.821900
