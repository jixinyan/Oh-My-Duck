"""PPO configuration for checking training plumbing, not a walking recipe."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DiagnosticRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 0
    num_steps_per_env = 24
    max_iterations = 5
    save_interval = 1
    experiment_name = "omd_isaac_pd_diagnostic"
    logger = "tensorboard"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1))
    critic = RslRlMLPModelCfg(hidden_dims=[64, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=2, num_mini_batches=2, learning_rate=0.001,
        schedule="fixed", gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)
