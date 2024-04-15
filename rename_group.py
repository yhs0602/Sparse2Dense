import wandb

if __name__ == "__main__":
    api = wandb.Api()
    specific_run_names = ["maze-gen1bug->2"]
    for r in api.runs("jourhyang123/craftground-sb3"):
        if r.name in specific_run_names:
            print(f"Updating {r.name}")
            r.group = "hcrmaze-generalization2"
            r.update()
