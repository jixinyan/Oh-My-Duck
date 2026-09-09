"""Serializable KL feedback through SB3's supported callable learning rate.

SB3 still owns PPO and its target-KL early stop. Unlike RSL's per-minibatch
analytic-KL adjustment, this controller uses the previous rollout's logged
approximate KL at the next rollout boundary. This difference is intentional.
"""
from dataclasses import dataclass
import math
from stable_baselines3.common.callbacks import BaseCallback


@dataclass
class KLAdaptiveLearningRate:
    rate: float
    target_kl: float
    minimum: float = 1e-5
    maximum: float = 1e-2

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.rate,self.target_kl,self.minimum,self.maximum)):
            raise ValueError("Learning rate and KL settings must be finite and positive")
        if not self.minimum <= self.rate <= self.maximum:
            raise ValueError("Initial learning rate must lie inside adaptive bounds")

    def __call__(self, progress_remaining):
        return self.rate

    def update(self, kl):
        if not math.isfinite(kl):
            raise FloatingPointError("Nonfinite PPO KL divergence")
        if kl > 2 * self.target_kl:
            self.rate = max(self.minimum, self.rate / 1.5)
        elif 0 < kl < .5 * self.target_kl:
            self.rate = min(self.maximum, self.rate * 1.5)
        return self.rate


class KLFeedback(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_start(self):
        schedule = self.model.learning_rate
        if isinstance(schedule, KLAdaptiveLearningRate):
            previous_kl = self.logger.name_to_value.get("train/approx_kl")
            if previous_kl is not None:
                self.logger.record("controller/previous_kl", float(previous_kl))
                self.logger.record("controller/next_learning_rate", schedule.update(float(previous_kl)))
