# SB3 critic repair and RSL cross-simulator diagnosis

Follow-up to the [framework review](rl-framework-status-2026-09-09.md).
User requested SB3 repair, diagnosis of the weaker MuJoCo RSL run, and capability
validation of the Newton RSL policies including MuJoCo transfer.

## Critic interface implementation

Fresh SB3 training now defaults to separate official actor/critic observation
groups. Native SB3 PPO uses its DictRolloutBuffer and its existing timeout
bootstrapping. Independent VecNormalize statistics are saved for each group.
The policy keeps 61 actor inputs, 14 outputs and task-local network widths;
critic observations cannot enter actor hidden layers or its exported graph.
Both terminal groups are sampled before reset without advancing live delay
buffers or consuming the training RNG. Export still uses official run_export
and the runner export method, with actor normalization baked in.

`--critic-observations actor` preserves the former input for controlled studies.
Resume infers the recorded layout (legacy artifacts imply actor-only) and rejects
changing layout while restoring a checkpoint. Old exports remain supported.
PPO optimizer, value loss and learning-rate defaults are unchanged in this
isolated interface change. This does not claim that critic repair alone fixes
learning behavior.

CPU checks: 14 SB3 tests passed, including privileged-input/gradient isolation,
native DictRolloutBuffer training and normalized terminal-critic bootstrapping,
model/normalizer save-reload, asymmetric actor graph parity, legacy graph parity,
and real observation-manager delay/history preservation. Required native
64-env/5-update, export and CPU/BAM video gates are pending.

## Frozen-policy transfer experiment

`outputs/diagnostics/newton-transfer-0909-02` uses immutable source `668b661`:
Newton StandUp checkpoint 13000 and Walking 14000, plus final MuJoCo RSL StandUp.
Native backends test recovery at seeds 42/100/101; CPU/BAM tests recovery at 42
and 100–115. Seed 42 includes videos. The same ONNX bytes are checked in each
backend. This avoids conflating simulator transfer with another training run.

Attempt 01 failed before any rollout because the diagnostic driver globally
selected OSMesa before loading its optional library. Attempt 02 keeps EGL in
the parent and selects OSMesa only in the existing isolated MuJoCo video worker;
both attempts and their logs are retained. No training physics changed.
Results and remaining causal uncertainty will be recorded after completion.
