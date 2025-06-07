import gymnasium
import numpy as np


class NormalizedPBIMWrapper(gymnasium.Wrapper):
    """
    Gymnasium Wrapper for Normalized Potential-Based Intrinsic Motivation (PBIM).

    Parameters:
        env (gymnasium.Env): Base environment.
        intrinsic_reward_fn (callable): Function f(obs, action, next_obs) -> float raw intrinsic reward F_t.
        gamma (float): Discount factor used in shaping (gamma).
        running_mean_alpha (float): Smoothing factor for updating mean intrinsic reward F.
        max_episode_steps (int, optional): Maximum steps per episode. If None, uses env.spec.max_episode_steps.
    """

    def __init__(
        self,
        env,
        intrinsic_reward_fn,
        gamma=0.99,
        running_mean_alpha=0.001,
        max_episode_steps=None,
    ):
        super().__init__(env)
        self.intrinsic_reward_fn = intrinsic_reward_fn
        self.gamma = gamma
        self.running_mean_alpha = running_mean_alpha
        self.bar_F = 0.0
        self.max_steps = max_episode_steps or getattr(
            env.spec, "max_episode_steps", None
        )
        self.reset_episode_storage()

    def reset_episode_storage(self):
        # Storage for one episode
        self.raw_F = []  # raw intrinsic rewards F_t
        self.shaped_F = []  # normalized shaping rewards F'_t
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_episode_storage()
        return obs, info

    def step(self, action):
        obs, ext_reward, done, truncated, info = self.env.step(action)
        # Compute raw intrinsic reward
        raw_F_t = self.intrinsic_reward_fn(self.last_obs, action, obs)
        # Compute normalized intrinsic reward F_t^norm = F_t - bar_F
        F_norm = raw_F_t - self.bar_F
        # Append to raw and shaped lists
        self.raw_F.append(raw_F_t)
        self.shaped_F.append(F_norm)
        self.step_count += 1

        # If terminal or truncated -> last step shaping
        is_last = (
            done or truncated or (self.max_steps and self.step_count >= self.max_steps)
        )
        if is_last:
            N = len(self.shaped_F)
            # Compute backward correction for t = N-1
            last_correction = 0.0
            for n in range(0, N - 1):
                Fp = self.shaped_F[n]
                last_correction += -(self.gamma ** (n + 1 - N)) * Fp
            self.shaped_F[-1] = last_correction
            shaped_reward = self.shaped_F[-1]
        else:
            shaped_reward = F_norm

        # Update running mean bar_F with raw intrinsic reward
        self.bar_F += self.running_mean_alpha * (raw_F_t - self.bar_F)

        # Combine external and shaped intrinsic rewards
        total_reward = ext_reward + shaped_reward

        # Store obs for next step's intrinsic function
        self.last_obs = obs

        return obs, total_reward, done, truncated, info

    def seed(self, seed=None):
        try:
            return self.env.seed(seed)
        except AttributeError:
            self.env.action_space.seed(seed)
