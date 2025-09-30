from tqdm import tqdm
import wandb
from download_s2d_trajectories import get_run


def main():
    # update timing_str for each run
    api = wandb.Api(timeout=120)
    WANDB_ENTITY = "jourhyang123"
    WANDB_PROJECT = "Sparse2Dense-room_experiments"
    run_id_to_timing_str = {}
    for seed in [0, 42, 9876, 7777, 2024, 1234]:
        for transition_timing in [1, 2, 3, 4, 5]:
            for timing in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                run_id = get_run(seed, transition_timing, timing)
                # 1m 2m 3m 4m 5m
                timing_str = f"{transition_timing}m"
                run_id_to_timing_str[run_id] = timing_str

    for run_id in tqdm(run_id_to_timing_str.keys()):
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        run.config["timing_str"] = run_id_to_timing_str[run_id]
        run.update()


if __name__ == "__main__":
    main()
