"""Construct a task recipe without choosing a simulation runtime or PPO learner."""

def build_environment(binding, *, play=False):
    cfg = binding.environment.build(play=play)
    if binding.robot_variant is not None:
        from oh_my_duck.rl.tasks.shared.backlash import make_backlash_variant
        cfg = make_backlash_variant(cfg, binding.robot_variant.build())
    return cfg
