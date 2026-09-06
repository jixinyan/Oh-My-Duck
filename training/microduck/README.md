# Microduck task and policy package

This is maintained Oh My Duck source, derived from Pollen Robotics' Apache-2.0 implementation at `29e887ecfbf5d37144759e5a9f8a176dfb83d547`. It is installed as `oh-my-duck-microduck` and imported as `omd_microduck`. `UPSTREAM.json` records file ancestry and import-time hashes; `LICENSE` preserves the original license. Edit these files directly. Cached research checkouts are not runtime task sources.

| Change | Source under `src/omd_microduck/` |
|---|---|
| Task registration | `configs/tasks.json` (project root) |
| Walking observations, rewards, events, curricula, actor/critic configuration | `tasks/microduck_velocity_env_cfg.py` |
| StandUp recipe and actor/critic configuration | `tasks/microduck_standup_env_cfg.py` |
| Custom MDP implementations | `tasks/mdp/` (commands, observations, events, curricula, state and reward families) |
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
2. Register its ID and backend bindings in the root `configs/tasks.json`: lazy environment factory, native RSL configuration and runner references. The framework registry is simulator-independent; the mjlab plugin builds native configurations from it. Custom IDs do not need a MicroDuck substring.
3. `python omd.py tasks --all` reads that same registry. Regenerate the detailed inventory with `training/mujoco/catalog.py`; the inventory is evidence, not a second registration source.
4. Add semantic tests and run the 64-environment / 5-iteration, reward-sign, resume, normalized export and behavior gates. Add the task to `configs/training.json` only if it belongs in the representative acceptance scope.

Isaac's full Walking/StandUp binding is still pending. Shared local robot and BAM sources are wired into Isaac, but adding a MuJoCo task does not yet register an Isaac task. The remaining binding must explicitly implement sensor, event, contact, DR and reset semantics; unsupported terms must fail visibly.

The imported inventory retains other recipes for future extension; only Walking and StandUp are selected for current reproduction. Task registration and CLI discovery now share one framework registry. Isaac task bindings remain explicitly absent until implemented.

## Checks

Run lightweight interfaces separately from simulator tests (they assert heavy modules are absent):

```bash
python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES='' .envs/mujoco-sb3/bin/python -m pytest training/microduck/tests training/mujoco/tests -q
CUDA_VISIBLE_DEVICES='' .envs/isaac-newton/bin/python -m unittest discover -s training/isaac_newton/tests
```

W&B is offline only. Policy publishing is not part of setup or validation; local package checks use `training/mujoco/package_policy.py`.
