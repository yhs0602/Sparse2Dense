# 새로운 WandB 실행을 시작하거나 기존 실행을 재개
import wandb

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT


def main():
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/z3juttf2")
    data = run.history()  # 모든 로깅 데이터를 포함하는 DataFrame을 반환
    # Save it as csv file
    data.to_csv("data.csv")


if __name__ == "__main__":
    main()
