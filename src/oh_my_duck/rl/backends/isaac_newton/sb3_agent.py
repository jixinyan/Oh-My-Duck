"""SB3 PPO diagnostic configuration, separate from environment and RSL-RL config."""


def diagnostic_ppo():
    return {
        "seed": 42, "policy": "MlpPolicy", "n_timesteps": 1920,
        "n_steps": 24, "n_minibatches": 2, "n_epochs": 2,
        "learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 1.0, "max_grad_norm": 1.0,
        "policy_kwargs": {"activation_fn": "nn.ELU", "net_arch": [64, 64], "log_std_init": -2.302585},
        "normalize_input": True, "normalize_value": False, "clip_obs": 100.0,
        "device": "cuda:0",
    }
