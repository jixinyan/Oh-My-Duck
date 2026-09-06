# Add a task or customize a policy

Task behavior belongs to `src/oh_my_duck/rl/tasks` and `rl/mdp`; robot models and
motors belong to `robotics/microduck`. The source is editable and versioned here.
Only generic simulator and native PPO packages are external dependencies.

1. Start from the closest maintained task factory and preserve its observation,
   noise, delay, BAM, randomization and NaN-guard stack.
2. Add observations, rewards, events, commands or curricula in the corresponding
   `rl/mdp` module. Use entity APIs for state writes and name-resolved joint/body
   indices. Keep all 61 actor slots and the 14 named servo outputs.
3. Register a task in `configs/tasks.json`. Each supported backend supplies an
   environment recipe, runtime factory, native RSL configuration and runner.
   Unsupported bindings remain absent. A runtime factory receives `cfg`, `task`,
   `device`, and `render_mode`; adding a task does not require editing a learner.
4. Configure actor/critic widths or native RSL classes in the task's `rsl_config`.
   For SB3, set `policy_configs.sb3` to a `module:factory` reference returning
   `Sb3PolicyCfg`. It receives `task_id` and the task's `agent_cfg`; reference
   `kwargs` may supply extra parameters. Return an editable native
   `ActorCriticPolicy` subclass when required. PPO itself remains native.
5. Verify configuration and physical assumptions, then run locally with 64 environments /
   5 iterations. Audit every weighted penalty, native resume and timeout state,
   normalized ONNX export, and deployment rehearsal before scaling training. Submit a job only for multi-GPU experiments.

The `omd tasks --all` inventory includes recipes outside the two representative
validation tasks. Registration is not evidence of learned behavior. Newton bindings
must pass actual motor/contact/reset/sensor checks; a PD diagnostic cannot stand in
for a task. See [official invariants](../third_party/microduck_rl/UPSTREAM_GUIDELINES.md).
