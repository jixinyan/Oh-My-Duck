"""Instantiate only the runtime explicitly registered for the selected task."""

def create_environment(task,cfg,*,backend,device,render_mode=None):
    binding=task.binding(backend)
    if binding.runtime is None:
        raise ValueError(f'Task {task.id} has no runtime factory for {backend}')
    return binding.runtime.build(cfg=cfg,task=task,device=device,render_mode=render_mode)
