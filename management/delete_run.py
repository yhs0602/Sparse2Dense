import wandb

if __name__ == "__main__":
    api = wandb.Api(timeout=60)
    for run_path in [
        "jourhyang123/craftground-sb3/crtuuo73",
        "jourhyang123/craftground-sb3/hindtb15",
    ]:
        try:
            run = api.run(run_path)
            run.delete()
        except Exception as e:
            print(f"Error deleting run {run_path}: {e}")
            continue
