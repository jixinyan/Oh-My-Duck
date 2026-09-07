# Copyright 2026 Pollen Robotics. Apache-2.0.
# Oh My Duck: task-local native PPO configuration, values preserved.
from .environment import (
    NUM_STEPS_PER_ENV,
    PpoWithSymmetryCfg,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
)

MicroduckRollerStandUpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # le normaliseur DOIT être baké dans l'ONNX par export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Symétrie OFF : SYMMETRY_CFG est câblé pour l'ancien layout 51D et casse
        # sur le 61D (même situation que tous les envs v1.5+).
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="roller_standup",
    run_name="roller_standup",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=15_000,
)

