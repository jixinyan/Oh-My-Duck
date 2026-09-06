"""Native mjlab simulation runtime for project-owned task recipes."""

def make_environment(cfg, *, task, device, render_mode=None):
    from mjlab.envs import ManagerBasedRlEnv
    return ManagerBasedRlEnv(cfg,device=device,render_mode=render_mode)
