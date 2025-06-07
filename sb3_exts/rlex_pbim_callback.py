import wandb
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm


class RLeXplorePBIMWithOnPolicyRL(BaseCallback):
    """
    On-policy callback integrating RLeXplore with Normalized PBIM shaping.

    - Watches interactions for IR updates.
    - Computes raw intrinsic rewards vector at rollout end.
    - Applies normalization (F_t - bar_F) and running-mean update of bar_F.
    - Applies PBIM last-step correction (Eq.34) on normalized rewards.
    - Adds shaped rewards into rollout_buffer advantages & returns.

    Params:
        irs: intrinsic reward sampler (must implement watch() and compute(samples, sync))
        ir_scale: float scaling factor for intrinsic reward
        gamma: discount factor for PBIM shaping
        running_mean_alpha: smoothing for bar_F update
        verbose: verbosity
    """

    def __init__(self, irs, ir_scale, gamma=0.99, running_mean_alpha=0.001, verbose=0):
        super(RLeXplorePBIMWithOnPolicyRL, self).__init__(verbose)
        self.irs = irs
        self.ir_scale = ir_scale
        self.gamma = gamma
        self.running_mean_alpha = running_mean_alpha
        self.buffer = None
        self.bar_F = 0.0

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = model.rollout_buffer
        # initialize running mean of raw intrinsic rewards
        self.bar_F = 0.0

    def _on_step(self) -> bool:
        # watch for IR updates each step
        obs = self.locals["obs_tensor"]
        acts = th.as_tensor(self.locals["actions"], device=obs.device)
        rews = th.as_tensor(self.locals["rewards"], device=obs.device)
        dones = th.as_tensor(self.locals["dones"], device=obs.device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=obs.device)

        self.irs.watch(obs, acts, rews, dones, dones, next_obs)
        return True

    def _on_rollout_end(self) -> None:
        # prepare batch for computing raw intrinsic rewards
        obs = th.as_tensor(self.buffer.observations)
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        acts = th.as_tensor(self.buffer.actions)
        rews = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)

        # compute raw intrinsic rewards vector F_t
        raw_F = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=acts,
                rewards=rews,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )  # torch tensor shape (N,)
        raw_F = raw_F.cpu().numpy()  # numpy array

        # normalize: F'_t = F_t - bar_F
        norm_F = raw_F - self.bar_F
        # update running mean of raw F
        mean_raw = raw_F.mean()
        self.bar_F += self.running_mean_alpha * (mean_raw - self.bar_F)

        # apply PBIM last-step correction for t = N-1
        N = len(norm_F)
        if N > 0:
            correction = 0.0
            for n in range(N - 1):
                correction += -(self.gamma ** (n + 1 - N)) * norm_F[n]
            norm_F[-1] = correction

        # scale and add shaped rewards to buffer
        shaped = norm_F * self.ir_scale
        # advantages and returns are arrays, just broadcast add
        self.buffer.advantages[:N] += shaped
        self.buffer.returns[:N] += shaped

        # logging
        wandb.log(
            {
                "mean_raw_intrinsic_rewards": mean_raw,
                "bar_F": self.bar_F,
            }
        )

        # no need to clear buffer; SB3 will reset buffer on next rollout
