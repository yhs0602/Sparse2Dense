import wandb


def get_commit_sha(run_id, project_name="your-project", entity="your-entity"):
    api = wandb.Api()
    run = api.run(f"jourhyang123/craftground-sb3/kaqfvtd1")

    # GitHub commit SHA 정보 가져오기
    commit_sha = run.commit

    return commit_sha


if __name__ == "__main__":
    run_id = "your-run-id"  # 예시 Run ID
    project_name = "your-project"  # 예시 프로젝트 이름
    entity = "your-entity"  # 예시 엔티티 이름

    commit_sha = get_commit_sha(run_id, project_name, entity)
    print(f"The GitHub commit SHA for run {run_id} is: {commit_sha}")
