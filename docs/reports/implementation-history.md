# Historical implementation log

These dated notes preserve earlier stages and failures. They are superseded by
[the current status](../implementation-status.md) and [domain-refactor evidence](domain-refactor.md).

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
| Harness bridge | Native task interface implemented | `HarnessBridge` supplies `open`, `submit`, `status`, `wait`, `stop` and `close`; `NativeTaskClient` connects application services and voice to the pinned native server |
| Voice services | Interfaces only | VoiceDesign, confirmed profile store, ASR, TTS, audio devices |
| Episode recording | JSONL implementation tested | Single process; payload storage/replay still pending |
| Training backend registry | Implemented and tested | Unimplemented Newton cannot fall back to another engine |
| Shared command schedule / joint contract | Implemented and tested | Exact 50 Hz boundaries and invalid input rejection |
| Official MuJoCo training/export adapter | Walking smoke and official export passed | 64 environments / 5 iterations; finite scalar and penalty-sign audit, local official schema-2 packaging passed |
| CPU MuJoCo/BAM headless evaluator | Replay and video work; behavior gate fails | Official alpha policy completes 14 s without falling but barely tracks forward/yaw commands |
| Isaac Lab / Newton Microduck task | PD physics/video and short training verified | Raw asset checks, RSL-RL/SB3 smoke training, RSL-RL numerical export pass; BAM locomotion remains pending. See [worker report](isaac-rl-validation.md) |
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

See [framework validation](framework-validation.md) and [architecture](../architecture.md). Subsequent job results will include task IDs, configurations, logs, metrics and video links.

## Isaac reference review and Git workflow (2026-09-06)

Reviewed `kabilankb/isaaclab-microduck` at `4310fe0` without running its simulator. Static comparison confirms the same HOME names/order/values as our contract; asset parity, runtime joint order, BAM, delayed observations, environment compatibility and video remain unvalidated. The reference uses explicit PD and has inconsistent locomotion status descriptions; its results are not our baseline. See the [review and implementation sequence](isaac-newton-reference-review.md).

Local history already contained framework commit `bca0a90`; this investigation uses `docs/isaac-newton-reference-review`. Subsequent features use focused branches and commits per [development conventions](../development.md). No remote push has occurred.

## Environment status correction (2026-09-06)

The bootstrap/uv sync processes are no longer running. Reading installed metadata through `.envs/mujoco/bin/python` succeeds: Python 3.12.13, mjlab 1.3.0, torch 2.9.1, mujoco 3.10.0, mujoco-warp 3.8.1, warp-lang 1.12.0 and rsl-rl-lib 5.0.1. This supersedes the earlier downloading status. The installer exit result was not recovered in this check; package metadata presence is not a full dependency/import check or worker CUDA, physics, rendering, training or evaluation validation. No job results are available yet.

Git inspection found 54 tracked project files and no modified or untracked entries in either workspace path (both resolve to the same directory). The three cached upstream repositories were also clean. Environments, downloaded models and caches remain intentionally ignored. An editor/user-side untracked-file report is awaiting the actual file paths for diagnosis.

## MuJoCo queue and Isaac implementation (2026-09-06)

- MuJoCo probe `omd-probe-20260906-01`, queue ID `local-e0ea0a5e16cf`, was submitted for one GPU. Scheduler reports **Wait for project quota**; no worker results exist yet. This supersedes earlier statements that no job had been submitted.
- CPU-only official ONNX structure and 16 synthetic-input checks passed, input `[1,61]`, output `[1,14]`, finite actions. SHA256 `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`. Local evidence: `outputs/policy-contract-20260906-01/result.json`. No physics or locomotion inference follows from this check.
- User authorized Isaac development while MuJoCo waits. Branch: `feat/isaac-newton-integration`; main includes the earlier framework/review history.
- Two independent Isaac locks resolved, installation started. Diagnostic task/asset conversion/worker entry points are implemented and pending runtime checks. BAM and full locomotion task are not implemented.
- **11 lightweight tests pass**: the original eight plus reordered/passive joint mapping, explicit diagnostic-backend boundaries, and rejection of stale/missing generated assets. Compilation and CLI help pass. These are not Isaac runtime tests.

## Isaac installed-package checks (2026-09-06)

Newton training environment installation completed. Dependency checks pass for 156 packages; task configuration, Newton camera configuration, runtime imports and actual training CLI help pass on CPU. There are now **11 lightweight framework tests plus 4 installed-Isaac CPU tests passing**. The asset-conversion environment is still installing at this checkpoint. No GPU simulation/training/video or BAM migration result is claimed. See [environment evidence and fixes](isaac-environment-resolution.md).

## Earlier checkpoint: both environments installed (2026-09-06)

Both Isaac environment setups and dependency checks now pass; this supersedes the preceding conversion-installing status. The Isaac asset conversion job `omd-isaac-assets-20260906-01` (`local-d0faa6d27c42`) has been submitted for one GPU and is queued. MuJoCo probe remains queued for project quota. No GPU results exist yet. Remaining work starts with conversion/solver diagnostics, then BAM and official locomotion migration; see [the current integration guide](../isaac-newton.md).

## Current checkpoint: worker results (2026-09-06)

This supersedes the queued status above.

- MuJoCo probe `omd-probe-20260906-01` **succeeded**, task ID `8f5e9735-5fdc-4029-8801-8f41c416d836`. Worker evidence: one NVIDIA H200, CUDA/Warp device availability, finite CPU MuJoCo stepping and EGL image output. Evidence: `outputs/probe-20260906-01/environment.json`, `egl.png`, and `outputs/jobs/omd-probe-20260906-01/scheduler.log`. The probe does not execute MuJoCo-Warp GPU physics, a robot policy, or training.
- Isaac conversion `omd-isaac-assets-20260906-01` **failed**, task ID `e9fdca75-aeb9-4133-8c82-79341865dd51`. Omniverse Kit requested first-run EULA acceptance in a noninteractive worker and exited with `Unable to bootstrap inner kit kernel: EOF when reading a line`. Evidence: `outputs/jobs/omd-isaac-assets-20260906-01/scheduler.log`. No completed asset manifest or Newton simulation result exists. License acceptance has not been configured by this project.
- Next: address the conversion startup prompt, rerun conversion and Newton diagnostics; complete official MuJoCo policy replay and short training/export verification. BAM, full Isaac locomotion and sim2sim remain pending.

## Headless startup fix prepared (2026-09-06)

Added explicit `assets -- --accept-eula`, which sets NVIDIA's supported `OMNI_KIT_ACCEPT_EULA=YES` only in the conversion child environment. Missing acceptance fails before importing Kit; an existing verified asset can still be reused. Two mocked regression tests cover rejection before child launch and process-scoped acceptance with failure propagation; no license is accepted by the tests.

Automatic approval review rejected the attempted commit-and-submit command because explicit user agreement to the EULA was not provided. That command did not execute; no retry job was submitted and no license was accepted. Local implementation and documentation can proceed independently. Actual conversion retry requires explicit user agreement to the NVIDIA Omniverse EULA.

## License approved and conversion retried (2026-09-06)

The user explicitly agreed to the NVIDIA Omniverse EULA. Submitted `omd-isaac-assets-20260906-02` with `--accept-eula`, one GPU, source commit `430e8ae`, queue ID `local-d87e23d96653`. The dispatcher is reconciling the submission; no worker result is available at this checkpoint. This supersedes the approval blocker above. The local regression suite now has **13 passing tests**; the two new tests use mocked subprocesses.

## Newton worker diagnostics and RL framework extension (2026-09-06)

- Conversion retry `omd-isaac-assets-20260906-02` succeeded, task `e9ec42f9-1680-4831-86bf-48719990a2ce`; verified asset fingerprint `e158fea3b1b7edc26738ff877ef7bd1660ae56ac333c27c0c2fd8b5b3c694c3a`. EULA startup is resolved.
- Probe `omd-isaac-probe-20260906-01` failed at argument parsing because the submitted separator was misplaced; no simulation occurred. Corrected probe `omd-isaac-probe-20260906-02`, task `f525ac94-c18b-4fb7-aa85-425e448f46ed`, succeeded: 2 environments, 100 control ticks, Newton `SolverMuJoCo` on `cuda:0`, canonical joint/action mapping and raw asset checks passed. Evidence: `outputs/isaac-probe-20260906-02/`.
- Manual inspection found the initial video's camera pose was stale despite nonuniform frames. `69c3329` enables live camera pose updates and asserts rendered camera position; repeat validation pending. Existing physical checks remain valid, but that video is not accepted as a useful visual diagnostic. USD physics-material binding warnings, contact parity and BAM remain open.
- Per user steering, branch `feat/rl-framework-selection` introduces independent RL-framework selection. RSL-RL routes preserve the existing launch commands; SB3 PPO delegates to pinned Isaac Lab with an optional dependency extra. Actual SB3 install and task-aware CLI help pass; GPU learning and framework-native checkpoint lifecycle verification remain pending. Root suite: 17 passing tests. See [RL framework scope and boundaries](../rl-frameworks.md).

## Current checkpoint: Newton and two RL frameworks validated for diagnostics (2026-09-06)

This supersedes the previous camera/queue/training/export-pending checkpoints. EULA bootstrap is resolved. The local-ground/camera-corrected Newton probe passed 100 ticks and visual inspection. Both RSL-RL and SB3 PPO completed short GPU training. SB3's corrected adapter captured 192 exact pre-reset observations for 192 timeout truncations; native model and normalizer were saved. RSL-RL normalized ONNX export passed 16 Torch/ORT comparisons (maximum absolute error `3.5762786865234375e-07`).

The application now independently selects simulation with `--backend` and learning with `--rl-framework`; `omd frameworks` lists actual combination/operation status. 17 lightweight tests and 7 installed-runtime CPU checks pass. Focused commits are on `feat/rl-framework-selection`; source, docs and local ground asset are tracked, runtime outputs/environments remain ignored. Full evidence, task IDs, failures and artifact paths: [Isaac/RL validation](isaac-rl-validation.md).

Still pending: MuJoCo robot training/export verification and SB3 adapter; SB3 native resume/ONNX export; RSL-RL resume/policy replay; BAM locomotion, sim2sim and upper-level functionality. A PD smoke run is not a trained walking policy.

## Official-task verification and SB3 continuation (2026-09-06)

Full official AGENTS.md read; 199 official CPU regressions passed, 1 skipped. Official MuJoCo walking 64-env / 5-iteration train, official normalized ONNX export, finite reward/penalty audit, and schema-2 publisher dry-run passed. CPU alpha-policy replay saves video but command tracking is inadequate. MuJoCo SB3 now reuses the official task with independent locked dependencies: 64-env / 5-rollout smoke passed, and native resume continued from 7,680 to 15,360 steps with 768 timeout snapshots. Three CPU tests cover the real mjlab observation-manager cache/delay boundary, timeout semantics and native normalized SB3 actor. MuJoCo SB3 official-runner ONNX export passed with maximum error 9.54e-7. Newton BAM completed 600 physics substeps and a 14-second official-policy replay; full task migration and contact parity remain pending. Isaac probe/eval now defaults to configurable 1280×720; an actual 720p/25-fps worker video passed metadata and visual checks. See [official compliance evidence](official-rl-compliance.md).


## Newton collision and 720p replay checkpoint (2026-09-06)

`omd-newton-contact-20260906-08` succeeded: all five official collision hulls, all 15 collision relationships and both explicit foot-ground contact parameter sets passed actual compiled-model checks. The static plane is retained; explicit Newton contact pairs resolve its filter translation limitation. A 700-tick / 14-second official BAM policy replay completed with 1280×720 / 25-fps H.264 video and visual frame inspection. Command tracking remains inadequate; full Isaac walking task migration and training are still outstanding. Future ground-friction DR must update explicit pair parameters. Detailed failed attempts and final evidence are retained in the official compliance report.


## Representative RL reproduction scope (2026-09-06)

用户收敛验收范围为官方 Flat Walking 与 Flat StandUp，覆盖 MuJoCo/mjlab 与 Isaac Lab/Newton、RSL-RL 与 SB3，共 8 个组合。完整 33 项官方任务仅作扩展清单，不能把注册当作复现通过。各框架保留原生 PPO：RSL-RL 使用原生多 GPU 分布式学习，SB3 使用向量环境与跨 GPU 独立任务并行，不引入异步 actor–learner。吞吐通过实际测量决定 GPU/环境数量。W&B 为标配但只用 offline，现有线上账号不是用户账号，禁止上传或同步。当前已验证结果保持原有范围，新增组合仍需 smoke、恢复、导出、有效行为与 sim2sim 验收。任务选择和日志默认值集中在 `configs/training.json`，详细矩阵见 [RL reproduction](../rl-reproduction.md)。本地 main 已合并，新开发分支为 `feat/rl-task-reproduction`。


## 2026-09-06：任务源码归属调整

根据用户要求，Microduck 的 task/MDP、actor/critic 配置、机器人模型、BAM 扩展、runner、导出与 CPU 回放迁入 `src/oh_my_duck`，成为项目内可编辑源码。`UPSTREAM.json` 与 Apache-2.0 许可证保留官方来源；缓存仓库只作对照，不再提供 Microduck 运行时任务。MuJoCo 两种框架与 Isaac 的共享机器人/BAM 引用已切换；通用 mjlab、Isaac Lab、Newton 和原生 PPO 仍作为依赖。初始 33 项配置清单在命名空间变更之外与官方基线一致，验收仍只覆盖 Walking/StandUp。

这是源码归属迁移，不等于完整 Isaac 任务接入或有效策略训练完成。迁移后 GPU smoke、恢复、导出和 Isaac 新指纹资产重建仍待验证；此前训练结果属于旧入口。扩展入口及下一步见 [Microduck package](../rl-task-extension.md) 与 [迁移交接](owned-task-migration.md)。


### 2026-09-06：框架重构进行中

用户进一步明确：源码接管不足以完成集成，需要重构为框架能力。现新增核心层 `oh_my_duck.rl.training.tasks` 和唯一任务注册源 `configs/tasks.json`，CLI、SB3 和原生 mjlab 注册均从该源读取；支持不含 MicroDuck 的自定义任务 ID，后端绑定缺失时明确拒绝。原 7,000 余行 MDP 已拆为 commands、observations、events、curricula、terminations、state 与 reward families，兼容修补移入后端模块；226 个有效函数/类保持原定义逻辑，仅增加显式模块依赖。完整 Newton 任务绑定、GPU 重新验收和有效策略训练继续进行，不视为已完成。usage 限制按用户最新指令取消。

## 2026-09-06 — approved domain architecture refactor

业务实现统一进入 `src/oh_my_duck`，按 RL、agentic、robotics、perception、voice、experience、core 和 infrastructure 分层。训练依赖锁独立保存在 `environments/`；任务、MDP、机器人和策略配置均为项目源码。外部 harness 保持规划/记忆职责，尚未实现的适配器不声明可用。详见 [architecture.md](../architecture.md)。

迁移后轻量测试 21 项、MuJoCo/SB3/官方 MDP 与 manifest 测试 53 项通过；新训练入口和 Newton 任务适配仍需 GPU 验证。双任务 × 双后端 × 双框架的训练、恢复、导出与回放验收尚未全部完成。旧代码清理在对应验证通过后执行。

**目录重构验证更新**：MuJoCo 两个代表任务 × 两种原生 PPO 的短训练、恢复与归一化导出已通过。Newton 的完整任务桥接仍处于物理/MDP gate，尚未开放任务注册；不能据此声明双后端行为验收完成。源码快照、失败记录和细节见 [domain-refactor.md](domain-refactor.md)。

**执行方式更新（2026-09-06，用户最新指示）**：单 GPU 开发、验证和训练直接在开发机 headless 运行；涉及多 GPU 的实验再提交 job。此前已提交任务保留其独立证据记录。

**Newton 任务接入更新（2026-09-06）**：Walking/StandUp 已通过实际物理与 MDP 门槛并注册共用运行时；StandUp 使用项目内 Isaac manager 在 graph 捕获前精确编译官方接触规则。两个原生 PPO 共用任务入口，恢复、导出及行为验收继续按独立门槛记录。导出的官方 MuJoCo 元数据参考与策略训练后端分别标注。详见 `docs/reports/domain-refactor.md`。
