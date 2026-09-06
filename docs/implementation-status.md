# Implementation status

Updated: 2026-09-06. Current priority: **complete the whole-project framework, then implement functionality incrementally**. The overall product remains the Agentic Microduck Project Design.

| Component | State | Evidence / next implementation |
|---|---|---|
| Local Git, origin, ignores and project overview | Implemented | Remote authentication fails; local history is retained |
| Application composition and unified CLI | Implemented | `ApplicationServices`, `omd.py` / `oh_my_duck.cli`, `omd status` |
| Identity, sensor, task and episode contracts | Initial interfaces implemented | Simulation/real identity and evidence serialization tests pass |
| Robot execution backends | Interface only | Add simulated backend and later official runtime adapter |
| Skills and lifecycle | Interface only | Bounded motion, measured results, cancellation |
| Tool catalog | Registry implemented and tested | Unknown tools unsupported; duplicate names / wrong request IDs rejected |
| Active perception / target records | Interface only | Sensor-backed look/inspect and tracking |
| Joint / velocity / sequence policy adapters | Interface only | Concrete inference runtimes |
| Harness bridge | Interface only | Deterministic mock first; no alternative agent loop |
| Voice services | Interfaces only | VoiceDesign, confirmed profile store, ASR, TTS, audio devices |
| Episode recording | JSONL implementation tested | Single process; payload storage/replay still pending |
| Training backend registry | Implemented and tested | Unimplemented Newton cannot fall back to another engine |
| Shared command schedule / joint contract | Implemented and tested | Exact 50 Hz boundaries and invalid input rejection |
| Official MuJoCo training/export adapter | Implemented, worker validation pending | Isolated setup still downloading dependencies |
| CPU MuJoCo/BAM headless evaluator | Initial implementation, unvalidated | Run fixed command sequence and save EGL video after setup |
| Isaac Lab / Newton Microduck task | Reference review complete; implementation planned | [Pinned third-party source review](reports/isaac-newton-reference-review.md); compatibility, assets, BAM, joint mapping and export remain unvalidated |
| Sim2sim comparison | Planned | Requires both actual backends and measured baselines |
| Hardware | Unavailable / deferred | No physical-robot tests |

## Evidence

- `python -m unittest discover -s tests -v`: **8 tests passed** on the host Python 3.13.13. These are lightweight framework/contract tests; they do not validate locomotion.
- `python -m compileall -q src scripts training omd.py`: passed.
- Root CLI help, software-maturity status and scheduler dry-run work without simulator initialization.
- `python omd.py train --backend isaac-newton`: expected exit code 2, explicit unavailable message, no physics fallback.
- Pinned upstream checkouts and public official walking policy acquired. Policy hashes are saved beside downloaded artifacts.
- The full upstream semantic audit is **not complete**. Source inventories and targeted review do not substitute for it.
- No training or evaluation job has been submitted in this framework-first iteration.

See [framework validation](reports/framework-validation.md) and [architecture](architecture.md). Subsequent job results will include task IDs, configurations, logs, metrics and video links.

## Isaac reference review and Git workflow (2026-09-06)

Reviewed `kabilankb/isaaclab-microduck` at `4310fe0` without running its simulator. Static comparison confirms the same HOME names/order/values as our contract; asset parity, runtime joint order, BAM, delayed observations, environment compatibility and video remain unvalidated. The reference uses explicit PD and has inconsistent locomotion status descriptions; its results are not our baseline. See the [review and implementation sequence](reports/isaac-newton-reference-review.md).

Local history already contained framework commit `bca0a90`; this investigation uses `docs/isaac-newton-reference-review`. Subsequent features use focused branches and commits per [development conventions](development.md). No remote push has occurred.
