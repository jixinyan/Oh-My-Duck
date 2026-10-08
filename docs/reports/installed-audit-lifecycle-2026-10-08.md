# Installed command deadlines and retained execution evidence

`omd validate package-audit` runs installed commands from outside the source
checkout. `--command-timeout SECONDS` supplies an optional positive finite
per-command deadline. The default permits the complete saved-evidence computation.
The implementation is `src/oh_my_duck/validation/release/package.py`.

A companion `.commands.jsonl` file records each command start, configured deadline,
process exit or interruption, elapsed time, return code and captured output hashes.
Records are flushed after each event. Process errors and interruptions terminate
the audit. A passed result is written only after every required check passes.
Both completed output files and existing partial command logs reject reuse.

Actual validation used immutable source
`984d6ff554b7259d73a4ea99d6db3e24aedfd744`, a fresh wheel/source distribution and a
fresh independent installed environment. CUDA visibility was empty throughout.

| Actual check | Verified result |
| --- | --- |
| Default complete audit | 418 source/resource files, three licenses, 19 installed CLI calls and 38 command lifecycle records |
| Explicit 60-second deadline | 17 installed CLI calls and 34 command lifecycle records |
| Recorded ONNX and physical audit | 1208 actual recorded policy actions, 1220 serialized updates, zero maximum ONNX error and a passed complete five-motion physical audit |
| Actual subprocess deadline interruption | Nonzero process result, retained started/interrupted records and no passed result file |
| Reuse of interrupted output | Rejected; original command-log SHA256 unchanged |
| Invalid deadline arguments | Zero, negative, NaN and infinity rejected before output creation |

All artifacts are under `outputs/acceptance/installed-deadline-20261008-01/`.
`complete.json`, `complete.commands.jsonl`, `complete-policy.json` and
`complete-metric.json` retain the full installed audit and actual saved evidence.
`explicit-deadline.json` and its command log retain the timed check. The actual
interruption has `interrupted.log` and `interrupted.commands.jsonl`.
`independent-audit.json` independently checks lifecycle records, terminal states,
invalid arguments and preserved partial evidence. Build and installation logs
are retained in the same directory.

GPU acceptance, fresh Newton kernel execution and RL remain stopped. These
results cover installed commands and real saved CPU evidence.

| Artifact | SHA256 |
| --- | --- |
| Independent lifecycle audit | `e494994de9987d3b60c1178069ab1a4eb3282e94d1f592cf8a2e9b25bcaee9d1` |
| Complete installed audit | `b603183cf3cb807c1f25d1eb5763295c3e1845fd6cda050f3a394c0ffdeed11f` |
| Explicit-deadline audit | `b51ac44b2b43a1bad95e02871529d46ceee60bf21c4ad82360f8ec56612ed77e` |
| Wheel | `1972d250014049c265b9ae9ffcae0348b617db4233a97e4f5a3c4b00d1bfa73a` |
| Source distribution | `f85888ba7bf44776215ad549ca57e8b05fba03f1669e6f9284f165780b7d0610` |
