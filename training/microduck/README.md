# Microduck task and policy package

This is maintained Oh My Duck source, derived from Pollen Robotics' Apache-2.0 implementation at `29e887ecfbf5d37144759e5a9f8a176dfb83d547`. It is installed as `oh-my-duck-microduck` and imported as `omd_microduck`. `UPSTREAM.json` records file ancestry and import-time hashes; `LICENSE` preserves the original license. Edit these files directly. Cached research checkouts are not runtime task sources.

| Change | Source under `src/omd_microduck/` |
|---|---|
| Task registration | `tasks/__init__.py` |
| Walking observations, rewards, events, curricula, actor/critic configuration | `tasks/microduck_velocity_env_cfg.py` |
| StandUp recipe and actor/critic configuration | `tasks/microduck_standup_env_cfg.py` |
| Custom MDP implementations | `tasks/mdp.py` |
| RSL-RL runner extension | `rl/rsl_runner.py` |
| SB3 policy class and policy kwargs | `rl/sb3.py` |
| Robot, HOME, collision and motor configuration | `robot/microduck_constants.py` |
| MJCF and mesh assets | `robot/microduck/` |
| BAM friction and backlash implementation | `actuator/friction_dr_bam.py` |
| Normalized runner export | `export.py` |
| Schema-2 manifest and publisher | `publish/` |
| CPU BAM rehearsal | `rehearsal/infer_policy.py` |

The initial recipes retain official behavior. Native `mjlab`, RSL-RL, SB3, Isaac Lab, Newton and BAM libraries remain pinned dependencies. Keeping native PPO does not require keeping Microduck task code outside this repository. The RSL configuration owns actor/critic model settings, including model class selection; SB3's factory can return a custom native `ActorCriticPolicy` subclass. Any policy changes still need normalized export parity and the 61-observation / 14-action deployment checks.

## Add a task

1. Add a recipe module beside the closest existing recipe. Build fresh train/play configs from the local Walking or StandUp factory, then change reward functions/weights, observations, events, curricula and actor/critic configuration there.
2. Register the new task once in `tasks/__init__.py` with `register_mjlab_task`, its train/play configs, RL config and runner class. Native framework discovery loads this package through its `mjlab.tasks` entry point; no cached Microduck package is installed.
3. Regenerate `configs/official_tasks.json` with `training/mujoco/catalog.py`. The CLI inventory currently selects MicroDuck-named tasks, so retain `MicroDuck` in a new task ID. This inventory is not a validation claim.
4. Add semantic tests and run the 64-environment / 5-iteration, reward-sign, resume, normalized export and behavior gates. Add the task to `configs/training.json` only if it belongs in the representative acceptance scope.

Isaac's full Walking/StandUp binding is still pending. Shared local robot and BAM sources are wired into Isaac, but adding a MuJoCo task does not yet register an Isaac task. The remaining binding must explicitly implement sensor, event, contact, DR and reset semantics; unsupported terms must fail visibly.

The imported inventory retains other recipes for future extension; only Walking and StandUp are selected for current reproduction. Custom-task CLI registration and backend capability discovery will be consolidated as the full Isaac binding is implemented. Do not describe that future interface as complete.

## Checks

Run lightweight interfaces separately from simulator tests (they assert heavy modules are absent):

```bash
python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES='' .envs/mujoco-sb3/bin/python -m pytest training/microduck/tests training/mujoco/tests -q
CUDA_VISIBLE_DEVICES='' .envs/isaac-newton/bin/python -m unittest discover -s training/isaac_newton/tests
```

W&B is offline only. Policy publishing is not part of setup or validation; local package checks use `training/mujoco/package_policy.py`.
