from oh_my_duck.infrastructure.usd_runtime import verify_execution_runtime


def create_environment(task, cfg, *, backend, device, render_mode=None):
    verify_execution_runtime(backend)
    binding = task.binding(backend)
    if binding.runtime is None:
        raise ValueError(f"Task {task.id} has no runtime factory for {backend}")
    return binding.runtime.build(cfg=cfg, task=task, device=device, render_mode=render_mode)
