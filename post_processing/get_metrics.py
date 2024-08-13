import wandb

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

# evolution last
# average 3 values from the last point
# episodic,
# average reward ,
# Success rate
groups = [
    "v30-crossw2-sparse-2",
    "v30-crossw2-dense-0",
    "v30-crossw2-dense-2",
    "v30-crossw2-transition-3000000-1",
    "v30-crossw2-transition-3000000-2",
    "v30-crossw2-transition-3000000-0",
    "v30-crossw2-sparse-0",
    "v30-crossw2-dense-1",
    "v30-crossw2-sparse-1",
]


def main():
    api = wandb.Api()
    results = {}
    for group in groups:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": group})
        lengths = []
        rewards = []
        eval_episodes = []
        left_success_counts = []
        middle_success_counts = []
        right_success_counts = []
        total_success_counts = []

        for run in runs:
            global_history = run.history(keys=["global_step"], pandas=False)
            if len(global_history) == 0:
                continue
            global_step = global_history[-1]["global_step"]
            if global_step < 10000384:
                print(
                    f"Skipping run {run.name} from group {group} as it is not finished yet. {global_step}"
                )
                continue
            del global_history
            eval_history = run.history(
                keys=[
                    "eval/mean_ep_length",
                    "eval/mean_reward",
                    "eval_episode",
                    "eval_((13, 1, 6), (13, 1, 7))/success_count",
                    "eval_((7, 1, 1), (8, 1, 1))/success_count",
                    "eval_((8, 1, 12), (7, 1, 12))/success_count",
                ],
                pandas=False,
            )
            if len(eval_history) > 0:
                last_length = eval_history[-1]["eval/mean_ep_length"]
                last_reward = eval_history[-1]["eval/mean_reward"]
                last_eval_episode = eval_history[-1]["eval_episode"]
                left_success_count = eval_history[-1][
                    "eval_((7, 1, 1), (8, 1, 1))/success_count"
                ]
                middle_success_count = eval_history[-1][
                    "eval_((13, 1, 6), (13, 1, 7))/success_count"
                ]
                right_success_count = eval_history[-1][
                    "eval_((8, 1, 12), (7, 1, 12))/success_count"
                ]
                total_success_count = (
                    left_success_count + middle_success_count + right_success_count
                )
                lengths.append(last_length)
                rewards.append(last_reward)
                eval_episodes.append(last_eval_episode)
                left_success_counts.append(left_success_count)
                middle_success_counts.append(middle_success_count)
                right_success_counts.append(right_success_count)
                total_success_counts.append(total_success_count)
            else:
                print(f"!!!Run {run.name} from group {group} has no eval data.")

        if len(lengths) > 0:
            mean_length = sum(lengths) / len(lengths)
            mean_reward = sum(rewards) / len(rewards)
            mean_eval_episodes = len(eval_episodes) / len(eval_episodes)

            results[group] = {
                "mean_length": mean_length,
                "mean_reward": mean_reward,
                "mean_eval_episodes": mean_eval_episodes,
                "mean_eval_left_success_count": sum(left_success_counts)
                / len(left_success_counts),
                "mean_eval_middle_success_count": sum(middle_success_counts)
                / len(middle_success_counts),
                "mean_eval_right_success_count": sum(right_success_counts)
                / len(right_success_counts),
                "mean_eval_total_success_count": sum(total_success_counts)
                / len(total_success_counts),
            }

    print(results)


if __name__ == "__main__":
    main()
