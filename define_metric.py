from typing import List

import wandb

from cross_w2.cross_w2_env import Goal


def define_metrics(goals: List[Goal]):
    for goal in goals:
        wandb.define_metric(f"{goal.name}/success_count", step_metric="episode")
        wandb.define_metric(f"{goal.name}/time_took", step_metric="episode")
        wandb.define_metric(
            f"eval_{goal.name}/success_count", step_metric="eval_episode"
        )
        wandb.define_metric(f"eval_{goal.name}/time_took", step_metric="eval_episode")
    wandb.define_metric("episode/length", step_metric="episode")
    wandb.define_metric("episode/reward", step_metric="episode")
    wandb.define_metric("eval_episode/length", step_metric="eval_episode")
    wandb.define_metric("eval_episode/reward", step_metric="eval_episode")
    wandb.define_metric("eval/mean_reward", step_metric="eval_episode")
    wandb.define_metric("eval/mean_length", step_metric="eval_episode")
