import tempfile
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from oh_my_duck.rl.learners.sb3.learning_rate import KLAdaptiveLearningRate, KLFeedback
from test_asymmetric_policy import make_model


def test_bounded_kl_feedback():
    schedule=KLAdaptiveLearningRate(1e-4,.01)
    assert np.isclose(schedule.update(.04),1e-4/1.5)
    assert np.isclose(schedule.update(.001),1e-4)
    assert schedule.update(.01)==schedule.rate
    for _ in range(100):schedule.update(1.)
    assert schedule.rate==1e-5
    for _ in range(100):schedule.update(.001)
    assert schedule.rate==1e-2


def test_native_optimizer_uses_and_restores_callable_state():
    model,env=make_model()
    model.learning_rate=KLAdaptiveLearningRate(1e-4,.01)
    model._setup_lr_schedule()
    class InjectKL(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_start(self):self.logger.record('train/approx_kl',.04)
    # Injection runs before controller; real PPO still performs the update.
    model.learn(16,callback=[InjectKL(),KLFeedback()])
    rate=model.learning_rate.rate
    assert np.isclose(rate,1e-4/1.5**2)
    assert np.isclose(model.policy.optimizer.param_groups[0]['lr'],rate)
    with tempfile.TemporaryDirectory() as d:
        model.save(Path(d)/'model.zip')
        loaded=PPO.load(Path(d)/'model.zip',device='cpu')
        assert isinstance(loaded.learning_rate,KLAdaptiveLearningRate)
        assert loaded.lr_schedule(.5)==rate
        loaded.learning_rate.update(.04)
        assert np.isclose(loaded.lr_schedule(.5),rate/1.5)
    env.close()
