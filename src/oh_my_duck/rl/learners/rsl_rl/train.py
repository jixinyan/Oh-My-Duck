"""Framework-owned headless orchestration of native RSL-RL PPO and torchrunx."""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Literal

from oh_my_duck.core.paths import project_root
ROOT = project_root()
from oh_my_duck.infrastructure.tracking import settings
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlBaseRunnerCfg


@dataclass(frozen=True)
class TrainConfig:
    env: ManagerBasedRlEnvCfg
    agent: RslRlBaseRunnerCfg
    gpu_ids: list[int] | Literal['all'] | None = field(default_factory=lambda: [0])
    backend: str = 'mujoco'
    action_rate_delay_iterations: int = 0
    torchrunx_log_dir: str | None = None


def run_train(task_id, cfg, log_dir):
    import torch
    import time
    from oh_my_duck.infrastructure.provenance import source_provenance, validate_resume
    provenance = source_provenance()
    started = time.monotonic()
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml, get_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.backends.mujoco.registration import register_tasks
    register_tasks()
    os.environ['WANDB_MODE'] = settings()['mode']
    local_rank, rank = int(os.environ.get('LOCAL_RANK', 0)), int(os.environ.get('RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires an available CUDA GPU')
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(local_rank)
    configure_torch_backends()
    cfg.agent.seed += local_rank
    cfg.env.seed = cfg.agent.seed
    task = project_tasks(ROOT).get(task_id)
    binding = task.binding(cfg.backend)
    from oh_my_duck.rl.tasks.interventions import apply_action_rate_delay, validate_intervention_resume
    intervention = apply_action_rate_delay(cfg.env, task=task_id, iterations=cfg.action_rate_delay_iterations,
                                           steps_per_iteration=cfg.agent.num_steps_per_env)
    print(f'[INFO] Training task={task_id} backend={cfg.backend} device={device} seed={cfg.agent.seed} rank={rank}', flush=True)
    from oh_my_duck.rl.training.runtime import create_environment
    env = create_environment(task,cfg.env,backend=cfg.backend,device=device)
    try:
        native_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
        agent_cfg = asdict(cfg.agent)
        if rank == 0:
            dump_yaml(log_dir / 'params/env.yaml', asdict(cfg.env))
            dump_yaml(log_dir / 'params/agent.yaml', agent_cfg)
        agent_cfg['omd'] = {'task':task_id, 'backend':cfg.backend, 'provenance':provenance, 'training_intervention':intervention}
        runner = binding.runner.resolve()(native_env, agent_cfg, str(log_dir), device)
        runner.add_git_repo_to_log(__file__)
        resume = None
        resume_progress = None
        if cfg.agent.resume:
            import yaml
            from oh_my_duck.rl.learners.rsl_rl.progress import checkpoint_progress
            resume = get_checkpoint_path(log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint)
            previous = json.loads((Path(resume).parent/'run.json').read_text())
            validate_resume(previous,task=task_id,backend=cfg.backend,framework='rsl-rl')
            validate_intervention_resume(previous, intervention)
            if previous.get('provenance',{}).get('upstream',provenance['upstream']) != provenance['upstream']:
                raise ValueError('Resume upstream pins differ')
            saved_agent = yaml.full_load((Path(resume).parent/'params/agent.yaml').read_text())
            if saved_agent['num_steps_per_env'] != cfg.agent.num_steps_per_env:
                raise ValueError('Resume rollout length differs from the saved native agent')
            print(f'[INFO] Loading native checkpoint {resume}', flush=True)
            infos = runner.load(str(resume))
            resume_progress = checkpoint_progress(
                runner.current_learning_iteration,
                infos['env_state']['common_step_counter'],
                cfg.agent.num_steps_per_env,
            )
            if env.common_step_counter != resume_progress['common_step_counter']:
                raise ValueError('Native runner did not restore the saved environment step counter')
            runner.current_learning_iteration = resume_progress['next_iteration_label']
        before = runner.current_learning_iteration
        if rank == 0:
            # Native RSL checkpoints already include curriculum state. Retain
            # identity before training so a periodic checkpoint can resume even
            # if the process never reaches the completion report below.
            (log_dir / 'run.json').write_text(json.dumps({
                'task': task_id, 'backend': cfg.backend, 'framework': 'rsl-rl',
                'status': 'running', 'iteration_before': before,
                'num_envs_per_rank': cfg.env.scene.num_envs,
                'num_steps_per_env': cfg.agent.num_steps_per_env,
                'world_size': int(os.environ.get('WORLD_SIZE', 1)),
                'provenance': provenance, 'behavior': 'unvalidated', 'training_intervention': intervention,
                'resume_progress': resume_progress,
            }, indent=2) + '\n')
        runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
        expected_steps = (before + cfg.agent.max_iterations) * cfg.agent.num_steps_per_env
        if env.common_step_counter != expected_steps:
            raise ValueError(f'Native environment progress differs: {env.common_step_counter} != {expected_steps}')
        if rank == 0:
            import subprocess
            (log_dir / 'run.json').write_text(json.dumps({
                'task': task_id, 'backend': cfg.backend, 'framework': 'rsl-rl',
                'num_envs_per_rank': cfg.env.scene.num_envs,
                'num_steps_per_env': cfg.agent.num_steps_per_env,
                'world_size': int(os.environ.get('WORLD_SIZE', 1)),
                'iteration_before': before, 'iteration_after': runner.current_learning_iteration,
                'completed_iterations_after': env.common_step_counter // cfg.agent.num_steps_per_env,
                'resume': str(resume) if resume else None, 'wandb_mode': settings()['mode'],
                'project_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                'behavior': 'unvalidated', 'provenance':provenance, 'training_intervention': intervention,
                'resume_progress': resume_progress, 'wall_time_s':time.monotonic()-started}, indent=2)+'\n')
    finally:
        env.close()
        if rank == 0:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def main():
    import tyro
    import mjlab
    from mjlab.utils.gpu import select_gpus
    from oh_my_duck.rl.training.tasks import project_tasks
    from oh_my_duck.rl.backends.mujoco.registration import register_tasks
    from oh_my_duck.rl.tasks.recipes import build_environment
    register_tasks()
    tasks = project_tasks(ROOT)
    task_id, remaining = tyro.cli(tyro.extras.literal_type_from_choices([t.id for t in tasks.list()]),
        add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS)
    # Shared recipe default. Unsupported backend bindings still fail before creating physics.
    task = tasks.get(task_id)
    backend = next((arg.split('=', 1)[1] for arg in remaining if arg.startswith('--backend=')), 'mujoco')
    if '--backend' in remaining:
        backend = remaining[remaining.index('--backend') + 1]
    binding = task.binding(backend)
    cfg = tyro.cli(TrainConfig, args=remaining,
        default=TrainConfig(build_environment(binding), binding.rsl_config.build()), config=mjlab.TYRO_FLAGS)
    task.binding(cfg.backend)
    os.environ['WANDB_MODE'] = settings()['mode']
    selected, count = select_gpus(cfg.gpu_ids)
    if not count:
        raise RuntimeError('No training GPUs selected')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, selected))
    os.environ['MUJOCO_GL'] = 'egl'
    suffix = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+cfg.agent.run_name
    log_dir = ROOT / 'logs/rsl_rl' / cfg.agent.experiment_name / suffix
    log_dir.mkdir(parents=True, exist_ok=False)
    if count == 1:
        run_train(task_id, cfg, log_dir)
    else:
        import torchrunx
        logging.basicConfig(level=logging.INFO)
        os.environ['TORCHRUNX_LOG_DIR'] = cfg.torchrunx_log_dir or str(log_dir/'torchrunx')
        torchrunx.Launcher(hostnames=['localhost'], workers_per_host=count, backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ('MUJOCO*','WANDB_MODE','WANDB_PROJECT',
                'WANDB_DIR','WANDB_RUN_GROUP','OMD_PROJECT_ROOT','PYTHONPATH')).run(run_train, task_id, cfg, log_dir)


if __name__ == '__main__':
    main()
