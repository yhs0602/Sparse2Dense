# usage:
# python wandb_runs_checker.py --group <group_name>
# Then it prints out the runs of the group;
# run id, seed, settings, status, and a link to the run.

import argparse
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True)
    args = parser.parse_args()

    runs = wandb.Api().runs()
    for run in runs:
        if args.group in run.name:
            print(
                run.id,
                run.config["seed"],
                run.config["settings"],
                run.state,
                f"https://wandb.ai/{run.id}",
            )


if __name__ == "__main__":
    main()
