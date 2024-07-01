import os

import wandb
from tqdm import tqdm
from wandb.apis.public import Run, File


def download_checkpoint(run: Run, download_dir) -> str:
    # check if the file already exists
    new_filename = os.path.join(download_dir, f"{run.group}-{run.id}.zip")
    if os.path.exists(new_filename):
        print(f"File {new_filename} already exists")
        return new_filename

    file: File = run.file("model.zip")
    file.download()
    # rename the file to the run id with group id
    os.rename(file.name, new_filename)
    return new_filename


def main():
    api = wandb.Api(timeout=120)
    runs = api.runs("jourhyang123/craftground-sb3")
    group_runs = [
        run
        for run in runs
        if run.group.startswith("v31-room-")
        and (
            run.group.endswith("sparse")
            or run.group.endswith("dense")
            or run.group.endswith("1000000")
            or run.group.endswith("2000000")
            or run.group.endswith("3000000")
        )
    ]
    print(f"Found {len(group_runs)} runs")
    # 해당 run들에 대해서, checkpoint를 다운로드.
    for run in tqdm(group_runs):
        download_checkpoint(run, "./checkpoints")


if __name__ == "__main__":
    main()
