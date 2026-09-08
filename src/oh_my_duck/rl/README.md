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
