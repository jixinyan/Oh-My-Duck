# Native acceptance source map

| Work | Public command | Implementation |
| --- | --- | --- |
| Check release configuration | `omd validate release-plan` | `release/plans.py` |
| Check remote metadata or run allocated-GPU stages | `omd validate release` | `release/campaign.py`, `release/worker.py` |
| Run declared metric cases | `omd validate metric` | `metric/campaign.py`, `metric/case.py` |
| Check pending motion requests and explicit command changes | `omd validate metric-admission` | `harness/metric_admission.py` |
| Measure independent goal evidence across retained tasks | `omd validate task-continuation` | `harness/task_continuation.py` |
| Audit saved physical measurements | `omd validate metric-audit` | `metric/verify.py` |
| Recompute saved official policy actions | `omd validate policy-audit` | `metric/policy.py` |
| Check registered training packages | `omd validate policy-registry` | `harness/registry.py` |
| Execute registered perpetual and episodic policies | `omd validate policy-packages` | `harness/package_execution.py` |
| Exercise physical motion boundaries | `omd validate motion-guard` | `harness/motion_guard.py`, `harness/control.py` |
| Measure official CPU head/body pose commands | `omd validate pose` | `harness/pose.py` |
| Audit pose commands and native physical responses | `omd validate pose-audit` | `harness/pose_records.py` |
| Audit native agent/task evidence | `omd validate replay` | `harness/replay.py` |
| Run model-driven navigation | `omd validate navigation` | `harness/navigation.py` |
| Audit saved navigation | `omd validate navigation-audit` | `harness/navigation_replay.py` |
| Audit an independent package installation | `omd validate package-audit` | `release/package.py` |
| Run model perception or audit original RGBD measurements | `omd validate perception` | `perception.py` |

Use the corresponding command with `--help` for required paths and parameters.
`omd replay` handles session open/task/status/export/close through
`experience/harness_replay.py`. Process ownership and cancellation belong to
`infrastructure/owned_process.py` and `infrastructure/acceptance_supervisor.py`.

Configuration modules import JSON Schema and motion declarations. Physical
execution loads the pinned native Harness and selected simulator explicitly.
Saved-evidence verification imports the libraries required to decode original
images and inspect or execute actual ONNX models. Install the `validation` extra
for saved-evidence audits; physical execution uses its locked runtime environment.

Each physical case preserves real source/model identity, observations, admitted
actions, execution counters, camera bytes, stopped samples and resource cleanup.
Metric provenance schema 2 verifies implementation and compatibility-entry-point
hashes against its recorded Git revision. Schema 1 retains its declared script
source verification. A passed metadata or saved-record check has its own scope;
current Newton execution, model-task acceptance and hardware require their own evidence.
