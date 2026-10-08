# Register training packages for native tools

`omd harness --policy-registry REGISTRY.json` extends the pinned official catalogue
with local training packages. The native worker exposes each registered alias
through `microduck.policy_catalog` and `microduck.select_policy`. The optional
`locomotion` mapping chooses the policy used by the parameterized `walk` and
`rotate` tools for each robot model.

Package admission requires a single `policy.onnx`, its schema-2 API-1 manifest,
61 actor observations, 14 canonical servo outputs, canonical HOME and graph
action scale 1.0. Graph metadata and actual batch-one CPU inference are checked.
The manifest must declare constant command encoding and zero idle twist. A
perpetual package has no duration; an episodic package has a positive duration.
Native policy execution currently accepts feed-forward graphs. API-2 recurrent
exports use the separate rehearsal implementation and require additional native
state evidence before registry admission.

The registry schema is version 1. Each entry contains `name`, `package_dir`,
`manifest_sha256`, `policy_sha256`, `command_channels`, and `robot_model`.
`package_dir` resolves relative to the registry file or uses an absolute path.
Aliases must be unique and preserve all official names. `command_channels`
explicitly declares any of `twist`, `head`, and `body`; an empty array declares
an autonomous constant-command skill. Select channels from the training task's
command semantics. Descriptive manifest text does not infer these channels.
`robot_model` is `allcollisions` or `groundcontact_rollers`.

`locomotion` maps either robot-model name to a registered or official alias.
Each selected policy must be perpetual, support twist, and match the robot model.
The defaults are `alpha_walking` and `roller`. The registry SHA256 becomes part
of the effective catalogue revision. Package metadata, training provenance and
validation status remain visible to the agent.

Check a completed registry in the locked CPU or simulation environment:

```bash
omd validate policy-registry --catalog OFFICIAL_POLICY_DIRECTORY \
  --registry REGISTRY.json --output outputs/acceptance/registry-new.json
```

Then pass the same file to the native deployment:

```bash
omd harness --policy-dir OFFICIAL_POLICY_DIRECTORY --policy-registry REGISTRY.json \
  --provider-config PRIVATE_PROVIDER_CONFIG --simulation-python LOCKED_RUNTIME_PYTHON \
  --edh-source PINNED_HARNESS_SOURCE
```

SSH simulation uses `--worker-policy-registry /absolute/REGISTRY.json` together
with the existing complete worker arguments. Package paths are evaluated on the
simulation host. The server forwards the registry identity to the native worker.
GPU execution retains the project's explicit allocation and acceptance rules.

Policy selection requires a confirmed native boundary and five measured stopped
samples. Episodic completion uses the manifest duration; `transition_policy`
preserves physical pose, velocity and last-action history. Registered policies
retain the same 50 Hz cadence, four BAM physics substeps, sensors, bounded
commands, ActionGate and motion guards. Robot mode and physical action scale
remain checked during selection.

The CPU package campaign measures registered perpetual execution, metric-tool
selection, complete episodic duration, transition to official standing control,
stopping and resource release:

```bash
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  omd validate policy-packages --catalog OFFICIAL_POLICY_DIRECTORY \
  --registry REGISTRY.json --scene-config configs/simulation-demo/apartment-metric.json \
  --edh-source PINNED_HARNESS_SOURCE --perpetual-policy WALKING_ALIAS \
  --episodic-policy STANDUP_ALIAS --output outputs/acceptance/packages-new
```

Provide the actual Mesa library directory through `LD_LIBRARY_PATH` when its
installation requires it. The campaign checks the initialized software renderer
and that CUDA remains uninitialized. It saves native events, observations, tool
results and original observer PNG frames. Saved actions can be independently
recomputed using the same registry:

```bash
omd validate policy-audit --directory outputs/acceptance/packages-new \
  --catalog OFFICIAL_POLICY_DIRECTORY --policy-registry REGISTRY.json \
  --output outputs/acceptance/packages-new-independent.json
```

Registry and native execution checks establish package integration. Learned
locomotion, recovery from every spawn, metric precision, model task verdicts,
Newton execution and hardware each require their own behavioral evidence.
