# Reinforcement learning

Choose the task first, then the simulation backend and native PPO framework.
Use `python omd.py tasks`, `train`, `campaign`, `preview`, `export`, `compare`, `rehearsal`
and `package` from the repository root. Generic simulators and PPO remain dependencies;
Microduck environment definitions, motors and policy contracts are owned source.

```text
rl/
├── tasks/
│   ├── walking/            environment.py + ppo.py
│   ├── stand_up/           environment.py + ppo.py
│   ├── sit_stand/          environment.py + ppo.py
│   ├── roller_walking/    environment.py + ppo.py
│   ├── …                  one directory per task family
│   ├── shared/            terrain, symmetry and backlash variants
│   ├── recipes.py         construct a registered recipe
│   └── catalog.py         inspect the complete task inventory
├── mdp/                   shared observations, rewards, resets and curricula
├── models/                framework-specific actor/critic builders
├── backends/              MuJoCo and Isaac/Newton simulation adapters
├── learners/              native RSL-RL and SB3 training/checkpoints
├── training/              task/runtime/framework registration
├── experiments/           campaigns, checkpoint recovery and video previews
├── artifacts/             normalization-aware export and local packaging
└── evaluation/            task batteries, sim2sim, CPU/BAM and video
```

## Change or add a task

Edit a family's `environment.py` for its reward composition, observations,
commands, resets and curricula. Edit `ppo.py` for actor/critic architecture and
native PPO defaults. Shared MDP functions live in `mdp/`; task-specific functions
can live alongside the family's configuration and be referenced directly.

Register factories in `configs/tasks.json`. Flat/rough/backlash variants reuse a
family with explicit arguments or a variant builder instead of duplicating source.
Simulator bindings, SB3 model factories, evaluation protocols and package profiles
are independent registration choices. Keep `__init__.py` lightweight; import the
explicit configuration module. All 33 registered variants remain available as an
inventory; only Walking and StandUp are accepted representative training targets.

`ppo.py` preserves native settings extracted from the official recipe; SB3 uses
its registered model factory with those settings where applicable. The PPO
configuration can read task toggles from `environment.py`. Do not import its own
`ppo.py` back into that environment. Simulator differences are explicit adapters,
not alternate reward definitions.

See the repository's `docs/rl-task-extension.md` and `docs/rl-campaigns.md` for
extension details, full training and checkpoint recovery. Registration and a
completed process do not prove that a policy learned the requested behavior.


For pinned-original reproduction controls, run `experiments/baseline_probe.py`
with each isolated environment's Python. It records compiled model arrays,
configs, resets and action traces; `--reference-reset` checks the exact original
reset function against owned Entity writes in a real MuJoCo environment. Original
sources must come from an isolated pinned archive, never a runtime cache import.
The export audit supports `--implementation official` only in an environment
without the owned package. Use Python `-P` when invoking its file directly to
avoid shadowing the installed MuJoCo package. These diagnostics do not certify
learned behavior. See the baseline audit report for the current controls.

## SB3 policy and optimization controls

Fresh SB3 runs consume separate registered actor/critic groups. The actor/export
contract remains 61 inputs and 14 servo actions; critic width follows the task
(StandUp 74, Walking 76). Edit `models/sb3.py` and `models/sb3_asymmetric.py` for
policy architecture; `learners/sb3/learning_rate.py` provides KL feedback through
native SB3's callable schedule. PPO itself remains the installed SB3 algorithm.

After `omd.py train --backend <backend> --rl-framework sb3 -- <task>`, use
`--critic-observations official|actor`, `--learning-rate-mode adaptive|constant`,
`--learning-rate <initial-rate>` and
`--initial-episode-phase randomized|synchronized` for explicit comparisons.
Fresh defaults are official/adaptive/randomized; resume preserves recorded
settings and rejects a changed critic layout. Legacy actor-only artifacts still
load and export. The corresponding snake_case fields in a campaign apply to
smoke, resume, capacity and full stages. See
`configs/experiments/sb3-official-critic.json` for a complete representative plan.
Native framework normalization, advantage/value loss and KL estimation still
have documented differences; shared rewards do not establish equal learning.
