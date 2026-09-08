"""Record a MuJoCo baseline in either an isolated official or owned environment.

Run this file directly with each environment's Python. The official control must
be installed from a pinned, isolated archive, never imported from upstream cache.
No task, physics, reward or learning defaults are changed by the probe.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def describe(value):
    """Stable config description; relocation is visible, not silently normalized."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: describe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    if callable(value):
        return {"callable": value.__module__ + "." + value.__qualname__}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def compare_ground_reset(env, reference, seed):
    """Compare the exact pinned function against Entity writes in one real runtime."""
    import ast
    import torch
    from oh_my_duck.rl.mdp import events
    from oh_my_duck.rl.mdp.state import _servo_joint_ids
    if '.cache/upstream' in str(reference.resolve()):
        raise ValueError('Use the isolated official source archive')
    source = reference.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'set_random_ground_state')
    # Only the reference function is executed. Its relocated globals/helpers are
    # identical in the pinned AST audit; there is no import of upstream tasks.
    namespace = dict(vars(events), _servo_joint_ids=_servo_joint_ids)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(reference), 'exec'), namespace)
    reference_function = namespace['set_random_ground_state']
    qpos, qvel = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    params = env.event_manager.get_term_cfg('set_ground_state').params.copy()
    params.update(sitting_joint_noise_std=0.1, sitting_tilt_max=0.2, face_up_roll_max=1.57)
    cases = []
    try:
        selections = (torch.arange(env.num_envs, device=env.device),
                      torch.arange(env.num_envs - 1, 0, -2, device=env.device),
                      torch.empty(0, device=env.device, dtype=torch.long))
        modes = ('mixed', 'face_down', 'face_up', 'sitting', 'standing')
        for selection, ids in enumerate(selections):
            for mode in modes:
                probabilities = {name + '_prob': (0.25 if mode == 'mixed' else float(name == mode))
                                 for name in modes[1:]}
                params.update(probabilities)
                results = []
                for function in (reference_function, events.set_random_ground_state):
                    env.sim.data.qpos.copy_(qpos)
                    env.sim.data.qvel.copy_(qvel)
                    torch.manual_seed(seed + selection)
                    function(env, ids, **params)
                    results.append((env.sim.data.qpos.clone(), env.sim.data.qvel.clone(),
                                    torch.cuda.get_rng_state().clone()))
                differences = [float((a.float() - b.float()).abs().max())
                               for a, b in zip(*results)]
                if any(differences):
                    raise AssertionError(f'Reset mismatch: selection={selection}, mode={mode}, errors={differences}')
                untouched = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
                untouched[ids] = False
                assert torch.equal(env.sim.data.qpos[untouched], qpos[untouched])
                assert torch.equal(env.sim.data.qvel[untouched], qvel[untouched])
                cases.append({'selection': selection, 'count': len(ids), 'mode': mode,
                              'max_qpos_qvel_rng_error': differences})
    finally:
        env.sim.data.qpos.copy_(qpos)
        env.sim.data.qvel.copy_(qvel)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
    return {'status': 'exact', 'reference_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'cases': cases, 'scope': 'state writes and CUDA RNG consumption, not learned behavior'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=('official', 'owned'), required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--reference-reset', type=Path, help='Pinned isolated official mdp.py; owned control only')
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        parser.error('Environment and step counts must be positive')
    if os.environ.get('WANDB_MODE') != 'offline':
        parser.error('WANDB_MODE must be offline')
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    torch.cuda.set_device(0)
    if args.implementation == 'owned':
        from oh_my_duck.rl.backends.mujoco.registration import register_tasks
        register_tasks()
    else:
        if importlib.util.find_spec('oh_my_duck') is not None:
            raise RuntimeError('Official control must not have the owned package installed')
        import mjlab_microduck
        if '.cache/upstream' in str(Path(mjlab_microduck.__file__).resolve()):
            raise RuntimeError('Use an isolated pinned archive for the official control')
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.envs import ManagerBasedRlEnv
    cfg, agent = load_env_cfg(args.task), load_rl_cfg(args.task)
    cfg.scene.num_envs, cfg.seed, agent.seed = args.num_envs, args.seed, args.seed
    (args.output / 'config.json').write_text(json.dumps({
        'env': describe(cfg), 'agent': describe(agent),
        'runner': describe(load_runner_cls(args.task)),
    }, indent=2) + '\n')
    metadata = {
        'implementation': args.implementation, 'task': args.task,
        'seed': args.seed, 'num_envs': args.num_envs, 'steps_per_stage': args.steps,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'wandb_mode': os.environ['WANDB_MODE'],
        'packages': {name: importlib.metadata.version(name) for name in
                     ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'torch', 'rsl-rl-lib', 'better-actuator-models')},
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status': 'running',
    }
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
        if args.reference_reset is not None:
            if args.implementation != 'owned':
                raise ValueError('Direct reset comparison requires the owned environment')
            metadata['reset_parity'] = compare_ground_reset(env, args.reference_reset, args.seed)
        model = env.sim.mj_model
        arrays = {}
        for key in dir(model):
            if key.startswith('_'):
                continue
            value = getattr(model, key)
            if isinstance(value, np.ndarray):
                arrays[key] = value.copy()
        np.savez_compressed(args.output / 'compiled_model.npz', **arrays)
        metadata['model_dimensions'] = {key: int(getattr(model, key)) for key in ('nq', 'nv', 'nu', 'nbody', 'ngeom')}
        stages = (0, 600, 1500, 2500, 4000) if 'StandUp' in args.task else (0,)
        generator = torch.Generator(device='cpu').manual_seed(args.seed + 10000)
        for stage in stages:
            env.common_step_counter = stage * 24
            obs, _ = env.reset(seed=args.seed + stage)
            trace = {}
            def capture(label, observations, reward=None, terminated=None, truncated=None):
                tensors = {'qpos': env.sim.data.qpos, 'qvel': env.sim.data.qvel,
                           **{'obs_' + k: v for k, v in observations.items()}}
                if reward is not None:
                    tensors.update(reward=reward, terminated=terminated, truncated=truncated)
                for key, value in tensors.items():
                    if value.is_floating_point() and not torch.isfinite(value).all():
                        raise FloatingPointError(f'{stage}/{label}/{key}')
                    trace[label + '/' + key] = value.detach().cpu().numpy().copy()
            capture('reset', obs)
            # Explicit noncontiguous reset exercises the rewritten event on a subset.
            ids = torch.arange(1, args.num_envs, 2, device=env.device)
            obs, _ = env.reset(env_ids=ids)
            capture('subset_reset', obs)
            for step in range(args.steps):
                action = torch.randn((args.num_envs, model.nu), generator=generator) * 0.15
                with torch.no_grad():
                    obs, reward, terminated, truncated, _ = env.step(action.to(env.device))
                capture(f'step_{step:04}', obs, reward, terminated, truncated)
            np.savez_compressed(args.output / f'trace_{stage:04}.npz', **trace)
            print(f'Completed {args.implementation} {args.task} stage={stage}', flush=True)
        metadata['status'] = 'completed'
    except Exception as error:
        metadata.update(status='error', error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        (args.output / 'result.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
