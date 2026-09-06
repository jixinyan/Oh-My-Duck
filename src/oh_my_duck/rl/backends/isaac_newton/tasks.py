"""Register only implemented diagnostic tasks, without loading scene configurations."""
TASK_ID = "Omd-Microduck-PD-Diagnostic-v0"


def register_tasks():
    import gymnasium as gym
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point="oh_my_duck.rl.backends.isaac_newton.environment:DiagnosticEnv", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "oh_my_duck.rl.backends.isaac_newton.config:DiagnosticEnvCfg",
                    "rsl_rl_cfg_entry_point": "oh_my_duck.rl.backends.isaac_newton.agent:DiagnosticRunnerCfg",
                    "sb3_cfg_entry_point": "oh_my_duck.rl.backends.isaac_newton.sb3_agent:diagnostic_ppo"})
