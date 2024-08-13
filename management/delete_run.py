import wandb

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

if __name__ == "__main__":
    api = wandb.Api()
    for run_id in [
        "fragrant-meadow-227",
        "gallant-resonance-223",
        "sparkling-thunder-220",
        "golden-shadow-215",
    ]:
        try:
            run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
            run.delete()
        except Exception as e:
            print(f"Error deleting run {run_id}: {e}")
            continue
