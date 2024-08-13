import wandb


def define_metrics(goals):
    for goal in goals:
        wandb.define_metric(f"{goal}/success_count", step_metric="episode")
        wandb.define_metric(f"{goal}/time_took", step_metric="episode")
        wandb.define_metric(f"eval_{goal}/success_count", step_metric="eval_episode")
        wandb.define_metric(f"eval_{goal}/time_took", step_metric="eval_episode")
    wandb.define_metric("episode/length", step_metric="episode")
    wandb.define_metric("episode/reward", step_metric="episode")
    wandb.define_metric("eval_episode/length", step_metric="eval_episode")
    wandb.define_metric("eval_episode/reward", step_metric="eval_episode")
    wandb.define_metric("eval/mean_reward", step_metric="eval_episode")
    wandb.define_metric("eval/mean_length", step_metric="eval_episode")
    wandb.define_metric("episode", step_metric="global_step")
    wandb.define_metric("eval_episode", step_metric="global_step")
