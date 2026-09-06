"""Register only implemented diagnostic tasks, without loading scene configurations."""
TASK_ID = "Omd-Microduck-PD-Diagnostic-v0"


def register_tasks():
    import gymnasium as gym
    if TASK_ID not in gym.registry:
        gym.register(id=TASK_ID, entry_point="omd_isaac.environment:DiagnosticEnv", disable_env_checker=True,
            kwargs={"env_cfg_entry_point": "omd_isaac.config:DiagnosticEnvCfg",
                    "rsl_rl_cfg_entry_point": "omd_isaac.agent:DiagnosticRunnerCfg"})
