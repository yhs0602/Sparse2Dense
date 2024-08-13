import wandb

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT


def main():
    api = wandb.Api()
    run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/z3juttf2")
    data = run.history()  # Returns a DataFrame containing all logging data
    # Save it as csv file
    data.to_csv("data.csv")


if __name__ == "__main__":
    main()
