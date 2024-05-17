from typing import List

import wandb

from cross_w2.cross_w2_env import Goal


def define_metrics(goals: List[Goal]):
    for goal in goals:
        wandb.define_metric(f"{goal.name}/success_count", step_metric="episode")
        wandb.define_metric(f"{goal.name}/time_took", step_metric="episode")
        wandb.define_metric(f"{goal.name}/success_rate", step_metric="episode")
        wandb.define_metric(
            f"eval_{goal.name}/success_count", step_metric="eval_episode"
        )
        wandb.define_metric(f"eval_{goal.name}/time_took", step_metric="eval_episode")
        wandb.define_metric(
            f"eval_{goal.name}/success_rate", step_metric="eval_episode"
        )
    wandb.define_metric("episode/length", step_metric="episode")
    wandb.define_metric("episode/reward", step_metric="episode")
    wandb.define_metric("episode/goal_idx", step_metric="episode")
    wandb.define_metric("eval_episode/length", step_metric="eval_episode")
    wandb.define_metric("eval_episode/reward", step_metric="eval_episode")
    wandb.define_metric("eval_episode/goal_idx", step_metric="eval_episode")
    wandb.define_metric("eval/mean_reward", step_metric="eval_episode")
    wandb.define_metric("eval/mean_length", step_metric="eval_episode")
    for v in [
        "time_took",
        "episode/reward",
        "success_rate",
        "success_count",
        "reached_goal",
    ]:
        wandb.define_metric(f"{v}", step_metric="global_step")
        wandb.define_metric(f"eval_{v}", step_metric="global_step")
        wandb.define_metric(f"eval_Earlystop_{v}", step_metric="eval_episode")
        wandb.define_metric(f"eval_Nostop_{v}", step_metric="eval_episode")
        wandb.define_metric(
            "eval_Earlystop/mean_time_took", step_metric="eval_group_idx"
        )
        wandb.define_metric("eval_Nostop/mean_time_took", step_metric="eval_group_idx")
        wandb.define_metric("eval_Earlystop/mean_reward", step_metric="eval_group_idx")
        wandb.define_metric("eval_Nostop/mean_reward", step_metric="eval_group_idx")
        wandb.define_metric("eval_Earlystop/success_rate", step_metric="eval_group_idx")
        wandb.define_metric("eval_Nostop/success_rate", step_metric="eval_group_idx")
    wandb.define_metric("eval_enabled_earlystop", step_metric="eval_episode")
    wandb.define_metric("eval_enabled_negative_reward", step_metric="eval_episode")
    wandb.define_metric("eval_group_idx", step_metric="eval_episode")
    for v in ["goal_idx", "time_took", "reward", "reached_goal", "success_rate"]:
        wandb.define_metric(f"{v}", step_metric="episode")
        wandb.define_metric(f"eval_{v}", step_metric="eval_episode")
