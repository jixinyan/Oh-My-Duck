# Implementation status

> 2026-09-30 场景与 policy 工具：原生 EDH 支持外部 Isaac 场景与显式 `transition_policy`。CPU MuJoCo/BAM 已验证 `kick_left` 的 25 个控制步、保留物理状态接续 `alpha_stand` 的 100 个控制步和 90 个连续停止样本。Office 的 2,291 个资源完成依赖校验，Newton/BAM 场景包含 3,646 个环境 collider。官方 `alpha_walking` 的行走、RGB、ToF 超量程状态已验证；native Harness 的量程内墙面检测与近障碍暂停也已完成：64 条射线命中原有墙面，central ToF 从初始 101 mm 降至 89 mm 时触发 `forward_proximity`。本次执行 227 个控制步、908 个物理子步，零命令阶段获得 85 个连续停止样本，外部障碍接触累计为零，机器人保持 upright。已生成 1080p 视频、ToF 热图、实际轨迹和距离/速度曲线，并完成产物检查。Isaac VLM/Verifier 正式导航、长距离指令响应与多场景 Demo 待验收。验证进程已关闭，RL 保持停止。见[近障碍停止与可视化](reports/isaac-proximity-2026-09-30.md)、[场景与 policy 进度](reports/isaac-scene-runtime-handoff-2026-09-30.md)与[场景资源](reports/isaac-scene-assets.md)。

> 当前核查（2026-09-30）：用户已停止 Walking 低速干预的全部八组配对训练及预览、评估进程；不自动恢复。固定训练源码为 `b8c36b2`，每组原定 8192 环境、50000 次更新，完整预算与最终行为验收均未完成。已有 checkpoint、输出与 W&B 文件保留，详见[停止记录](reports/rl-walking-user-stop-2026-09-30.md)。此前 MuJoCo/RSL-RL Walking 与 Newton/RSL-RL StandUp seed 42、43 的三组完整训练、导出和封装均已完成，最终结果均为 `behavior_failed`，详见[上一轮项目结果](reports/project-status-2026-09-29.md)。语音 HTTP 服务、Mac 内置扬声器播放、内置麦克风录制与合成期间停止后的无迟到播放已通过实际验证；合成原文为「你好，我是小鸭。我们现在检查语音连接。」、麦克风回读为「您好，我是小丫。我们现在检查语音连接。」，详见[语音交互验收](reports/voice-interaction-validation-2026-09-29.md)。原生 EDH 公寓会话已完成真实图像、工具、ActionGate 动作、`policy_stop` 和独立 Verifier 闭环；固定出生位置的独立 office 导航获得正式 passed，全程外部障碍接触累计 0，详见[公寓导航验收](reports/harness-office-navigation-2026-09-30.md)。项目目前没有可用真机。

> 2026-09-30 Walking 因果干预门禁完成时记录：针对保存策略在 `0.1 m/s` 近似静止、`0.2–0.4 m/s` 才响应的证据，新增的 `low_speed_tracking_boost` 配对实验完成 8 个 run 的 smoke、导出、恢复、8192 容量和 CPU/BAM rehearsal 门禁；默认值为 0，官方 Walking 配方不变。当时按最小目标在门禁后停止，尚未进入完整训练，未产生可封装 tool 的策略。首次门禁暴露并修复了 managed-host EGL loader 传播问题，Newton 资产按当前 fingerprint 重建后四组门禁通过。后续启动记录见[Walking 启动记录](reports/rl-walking-low-speed-launch-2026-09-29.md)，目前状态见[停止记录](reports/rl-walking-user-stop-2026-09-30.md)；门禁证据见[低速干预](reports/rl-walking-low-speed-intervention-2026-09-30.md)与[门禁记录](reports/rl-walking-low-speed-gates-2026-09-30.md)。

> 2026-09-30 ProtectiveFall 任务入口：已把完整碰撞资产、`servo_impact_contact`、伺服 stall/加速度保护项和 fallen smoothness scaling 组合为独立 `Mjlab-ProtectiveFall-Flat-MicroDuck` 配置，CLI 标为 `experimental_unvalidated`。官方 VelStand、Flat Walking 和 Flat StandUp 配置未被改写；该入口只通过静态配置/14-servo 编译测试，尚未训练、Newton 验证、导出、CPU/BAM 演练或封装为 tool。

> 2026-09-30 官方 develop/main 迁移续项：已把官方 API-2 显式状态 LSTM 的 `float32`/动态 batch/状态形状检查、失败清空状态和私有 reset 边界迁入发布校验；保留前馈 API-1。新增 deterministic BT.601/UYVY 相机、8×8 ToF 逐射线保护、官方 v15/alpha4/alpha16 足底 odometry anchor 数据，以及 protective-fall 的 servo stall/acceleration 与 fallen smoothness MDP term。对应 CPU 测试已通过；TCP body server、真实硬件 provisioning、完整 VelStand expert-BC 与 Newton GPU 行为门禁仍未完成，不能把这些接口当作已验证 locomotion tool。

> 2026-09-30 Harness 状态：`omd harness` 使用固定 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd` 原生运行时及独立 CPU 环境。真实 Astra 会话已接收 RGB 图像、ToF、IMU、关节与里程计，调用 MicroDuck 工具；官方 ONNX 策略的 14 关节动作经过 ActionGate，每次执行四个 0.005 秒 MuJoCo/BAM 子步。每条运动命令限定为 5–100 个实际控制步；真实 ToF、外部接触和位姿停滞可触发原生 Gate 暂停。独立新会话从 corridor 固定出生位置导航至 office，累计 760 个控制步与 3040 个物理子步，其中最后 100 个控制步使用零命令；确认 87 个连续停止样本，非地面外部接触累计 0。独立 Verifier 判定 `goal_reached=true`、连续达标 108/5 个控制步，Planner 调用原生 `tasks.finish`，run 终态 succeeded。取消、旧 lease 和关闭任务时在途请求的真实边界测试见[真实闭环记录](reports/harness-apartment-live-2026-09-30.md)；本次导航证据见[公寓导航验收](reports/harness-office-navigation-2026-09-30.md)。

> 2026-09-30 官方上游复核：已抓取 `pollen-robotics/microduck_rl` `develop@cfe1c2a` 与 `pollen-robotics/microduck` `main@f0d934e`，并更新 `configs/upstream.json`。已迁移通用的粗糙地形 reset 原点修复，避免把局部高度写成世界高度导致机器人出生在地形内部；新增 CPU 回归覆盖非零 terrain origin。上游 protective-fall/VelStand 的其余语义变化保留为独立后续变体，不改变代表性 Walking/StandUp 基线。详见 [官方上游复核](reports/upstream-review-2026-09-30.md)。

> 2026-09-30 自训练策略状态：jd_B300 的 MuJoCo/RSL Walking 50000 次更新策略在无推扰前进命令 `0.1 m/s` 下仅约 `0.3%` 响应；Newton RSL StandUp 两个 15000 次更新策略在俯卧和仰卧恢复上失败。训练、导出、回放和视频产物保留；这些自训练策略尚未作为已验收 locomotion tool 注册。当前 Harness 公寓接入使用有独立 manifest 来源的官方预训练策略，导航目标继续按原生 Verifier 验收。详见 [jd_B300 诊断](reports/rl-jd-interrupted-2026-09-26.md)。

> 2026-09-26 RL 状态：`jd_B300` 上的三组正式训练在完成预算前同时中断，原始 checkpoint 和日志均已保存；退出代码 247 的原因尚未确认。Walking SB3 保存于第 6000 次更新，原生 MuJoCo 无推扰评估完成 700 步，但前进与转向响应未通过评分；该策略目前保留。Newton RSL StandUp 两个 checkpoint 各已完成 5001 次真实更新，原生 Newton 无推扰评估在单一 seed 下均为 2/4 姿态。新的 MuJoCo/RSL Walking 官方配置对照已在 GPU6 进行完整训练；原生恢复计数修复通过真实 checkpoint 连续恢复验证，两组 Newton StandUp 已在 GPU2/GPU1 接续剩余 9999 次更新，训练日志均连续记录第 5001 至 5005 次更新。独立预览流程用真实 StandUp checkpoint 生成四份视频。已检查的 TensorBoard 标量均为有限值，现有训练奖励不能代替行为验收。详见 [中断训练诊断](reports/rl-jd-interrupted-2026-09-26.md) 与 [官方实现核对](reports/rl-official-comparison-2026-09-26.md)。

> 2026-09-23：`jd_B300` 的 CUDA 13 正式环境通过实际 GPU 检查，macOS 和 Linux 分别通过同组 55 项 CPU 测试。Isaac 资产转换通过，三组训练均完成启动检查、导出、恢复与容量检查，并已进入完整预算：Walking 使用 GPU 6，Newton StandUp 的 seed 42、43 分别使用 GPU 2、1。三组在线 W&B 记录已开始并出现正常更新；短训练回放行为未达标（返回码 2），最终行为验收需要完整训练与评估。语音与仿真执行仍待实现。GitHub 开放 Issues 和 Pull Requests 均为 0。见 [当前状态与未完成工作](reports/project-status-2026-09-23.md)。下方保留历史验证记录。

> 2026-09-21 CPU 验证：Walking 使用逐阶段评分版本 3；训练前检查复用覆盖完整源码、配置及依赖声明；SB3 按指定目录恢复并核对训练预算。53 项 CPU 测试通过，包括实际 Git 操作和原生 PPO 保存、加载、继续训练。Microduck GPU 仿真和历史策略重新评分尚未执行。见 [验证记录](reports/project-review-2026-09-21.md)。

> 2026-09-21 服务器迁移：按用户最终决定，仅迁移 Git 管理的源码、配置、锁文件与文档，合入并推送 main；checkpoint、normalizer、日志、视频、离线 W&B 和本地环境不上传。完整数据归档已取消，原训练产物保留在旧文件系统。原 Walking job 已不在调度 API 中，不自动恢复；既有策略结论保留为历史证据，不代表重新训练必然复现。见 [迁移交接与恢复说明](server-migration-2026-09-21.md)。

> 2026-09-14 策略复盘：231 份已完成周期预览均未通过完整行走标准。最近五个 checkpoint 中，MuJoCo SB3 官方课程、MuJoCo RSL 延后课程相对更值得做无推扰验证；Newton RSL 延后课程后期前进退步，Newton SB3 两组前进仍弱。延后平滑没有通用收益。横向瞬时误差包含快速摆动，不能直接等同持续侧滑；暂停状态和验收标准不变，本次仅分析既有视频与轨迹。见 [策略表现复盘](reports/rl-walking-policy-assessment-2026-09-14.md)。

> 2026-09-14 暂停状态：调度器将 `omd-walk-pacing-0913-01` 标记为 **Suspended**，八组日志均停止在 07:38 UTC 左右，本地 `running` 为滞后记录。暂停原因未返回，不能归因于代码或调度抢占。各组最后日志约 19803–41043 / 50000 更新；最近 checkpoint、normalizer、日志及视频保留，最终验收未执行。本次查询未重启或重新提交任务。见 [暂停与 checkpoint 记录](reports/rl-walking-suspension-2026-09-14.md)。

> 2026-09-13 完整训练迭代：前轮因果 job 已 Succeeded，32 个回放及两个 Newton 任务的全部门槛完成，四份周期导出与官方路径数值误差均为 0。无推扰对照确认 SB3 主动行走很弱；Newton RSL 在两后端的 play 配置下均有前进/转向响应，训练配置差异仍待定位。下一轮已提交单节点 8 GPU 的 Walking 配对完整训练（`omd-walk-pacing-0913-01`，已 Running，八组流水线已启动）：四种后端/框架组合，各比较官方任务课程和延后动作平滑课程，8192 环境、50000 更新；同一 job 内检查通过自动训练，定期视频与最终无推扰/跨后端/CPU 验收。该课程调整是待检验假设，不是已证实修复；StandUp 后续验收仍开放。见 [完整实验记录](reports/rl-walking-pacing-2026-09-13.md)。

> 2026-09-13 调度方式更新：后续常规 job 一次执行必要启动检查、完整训练预算及最终评估/视频；检查通过后自动继续，不再常规单独提交短验证 job。当前已在运行的 `omd-rl-causal-0913-01` 仍是原定短验证任务：32 个回放已完成，两任务周期导出已通过数值一致性检查，StandUp 最后容量检查尚在执行。

> 2026-09-13 诊断完成：`omd-rl-diagnose-0912-02` 的 49 个用例全部执行，不能等同于行为通过。Newton RSL 最终起身在两后端 seed=42 均 4/4；CPU/BAM 17 个种子中趴倒 14/17，其余三种姿态各 17/17。其他起身策略仍为 2/4，Walking 未完成验收；Newton Walking 有前进/转向能力，但横向摆动与跨后端前进失败需分开诊断。下一轮补齐只改变推扰强度的配对对照和视频，并验证 Newton 周期导出的真实训练回调；暂不盲目恢复完整训练。下一轮 `omd-rl-causal-0913-01` 已 Running（1 节点 4 张 H800，32 个回放对照后执行两任务短训练门槛）；24 项 CPU 测试与 9 个子测试通过。详见 [因果回放记录](reports/rl-causal-replay-2026-09-13.md)。以下为历史快照。

Updated 2026-09-12. RL iteration resumed by user instruction; single-node,
multi-GPU scheduler jobs are authorized. A 49-case frozen-policy diagnostic
matrix is prepared, including weighted reward traces and 17-seed CPU/BAM checks
of the final Newton StandUp policy. Diagnostic training-stage/push interventions
are explicitly ineligible for standard acceptance. Ten protocol/profile tests and three native export-metadata/callback tests
pass. Newton periodic export metadata indexing is repaired; GPU callback
integration remains pending. Diagnostic job `omd-rl-diagnose-0912-02` is accepted
(1 node, 4 GPUs) and waiting for project quota; attempt 01 failed on a shared lock.
See [current diagnosis](reports/rl-learning-diagnostics-2026-09-12.md). New tasks remain inventory entries
until their own validation, evaluation and packaging gates are implemented.
See [task catalog](rl-task-catalog.md). The pause below is historical.

Updated 2026-09-09. Product scope remains the Agentic Microduck Project Design.

**Training paused by user; zero project RL processes remain.** Five remaining
learners and both live preview workers were stopped with preserved checkpoints,
logs, videos and identity/hash records. No automatic resume is scheduled.
Newton RSL StandUp final passes 4/4 at seed 42 in Newton, MuJoCo and CPU/BAM;
final multi-seed robustness is open. MuJoCo RSL StandUp seed 43 completed with
2/4. Repaired SB3 has stable updates but incomplete learned behavior; Walking
also remains below acceptance. Next priority is causal debugging, not additional
long training. See [current state and debugging handoff](reports/rl-debug-handoff-2026-09-09.md)
for exact saved steps, evidence and the investigation sequence. CLI maturity
reflects partial behavior with training paused. All notes below are historical.

Latest operational update (2026-09-09): four old Walking learners were
intentionally stopped after repeated behavior failures: owned MuJoCo RSL, old
MuJoCo SB3, old Newton SB3, and the original 4096-env MuJoCo RSL control. All
artifacts and checkpoint hashes are preserved with intentional-stop records.
Six learners continue: Newton RSL Walking, the four repaired SB3 runs and the
MuJoCo RSL StandUp seed-43 control (now in full training). Newton RSL StandUp
finished 15000 updates and passed its final native 4/4 pose preview; final
transfer acceptance remains open. The paragraphs below are earlier snapshots.

Latest repair: native SB3 now has separate official critic observations (74D
StandUp / 76D Walking), serialized KL learning-rate feedback, randomized initial
episode phases and matching terminal/normalizer/export/resume paths. 29 tests
and 9 subtests pass. A paired 32-update continuation reduces mean KL from 0.073
to 0.012; this does not prove learned behavior. All four fresh 8192-env runs passed their individual gates and produced full PPO
updates in `sb3-repair-0909-01`, source `bea3eb1`; offline GPU processes verified.
Newton StandUp 13000 passes 12/12 scenarios in each native backend, but CPU/BAM
prone recovery passes only 3/17. No-push Walking 14000 retains forward/turn
response in both native backends; deployment forward motion fails. Details:
[repair and transfer](reports/sb3-critic-transfer-2026-09-09.md).
The SB3 repair milestone is merged and pushed to main `49b56de`; follow-up is
on `feat/rl-transfer-validation`. The complete 14-group frozen-policy battery
confirms MuJoCo StandUp final remains 9/12 in each backend (all supine failures).
Standard Walking with pushes fails acceptance in both backends; no-push results
are diagnostic only. The seed-43 MuJoCo RSL control (`1cb7738`, GPU 1) is running
fresh startup gates before its unchanged 8192-env/15000-update official recipe.
It has a background checkpoint preview worker starting at update 1000.
The following review and operational notes are earlier snapshots.

Latest behavior review (2026-09-09 02:29 UTC): five of the new 8192-env learners
continue, two MuJoCo StandUp runs completed (RSL 3/4, SB3 0/4), and Newton SB3
StandUp was intentionally stopped after ten consecutive 0/4 previews and
persistent excessive KL. Its update-10000 bundle is preserved and hash-verified.
Newton RSL StandUp repeatedly passes its native four-pose battery; final
sim2sim/CPU and multi-seed acceptance remain open. Newton RSL Walking responds
to forward/yaw commands but misses the lateral RMSE gate. Current SB3 critic
input is actor-only 61D versus official RSL's separate 74D input, alongside
native PPO/normalizer differences. Learner equivalence has not been established.
See [behavior and framework review](reports/rl-framework-status-2026-09-09.md).
The following September 8 operational paragraphs are historical snapshots.

Unused-process cleanup completed: all 26 processes in the old paused
`shared-gpu7-0908-01` tree exited, releasing 23.2 GiB on GPU 7. Current eight
learners, original controls and preview workers remain active; retained checkpoint
hashes are unchanged. See the fixed-size training report for the cleanup audit.

Latest user decision supersedes environment sweeps: all eight new combinations
use 8192 environments. Scaling and the old handover were explicitly stopped;
completed matching gates will be reused, with missing gates run independently.
`configs/experiments/representative-8192.json` keeps native PPO and task budgets,
SB3 learning rate 1e-4, and offline W&B. All eight full learners have produced PPO updates and passed startup checks;
each uses 8192 environments. Live process checks confirm offline W&B. Campaign `fixed-8192-0908-01`, source `3070b71`.
See [fixed-size launch evidence](reports/rl-fixed-8192-2026-09-08.md).

Latest scope: train both tasks across both owned backends and both native PPO
frameworks. `measured-env-0908-03` (source `f9fcf80`) is preparing on GPUs
1/6/2/3/4/5: six initial 64-env/5-update training smokes passed; Newton SB3
Walking/StandUp are queued for the next free card. All eight environment counts
remain to be selected from measured PPO throughput with 15% VRAM headroom.
Full training now starts per combination after its own gates pass. The old
global-barrier supervisor is paused while its live preparations continue.
`independent-full-0908-01` (source `392776a`) now adopts each completed
preparation independently; at startup, all new full learners still awaited their
own remaining gates. New campaigns release GPU slots during CPU stages. Original controls continue on 0/7. Their preview controller is temporarily
paused and will automatically resume after isolated calibration. Prior failed
startup artifacts are preserved. See [selection and startup evidence](reports/rl-environment-selection-2026-09-08.md).

Latest user preference: no continuous LLM training polling. A background worker
saves native videos every 1000 PPO updates (24000 control steps per environment)
and at the final checkpoint; unified CPU video/16-reset diagnosis runs after both
original learners end. Earlier intermediate milestone and paused-owned preview
watchers were replaced. Gallery: `outputs/previews/official-periodic-0908-01/index.html`.

Earlier official-first phase: original MuJoCo Walking/StandUp continue. The user
reopened idle GPUs: Walking remains on GPU 7; StandUp migrates from checkpoint
2000 to GPU 0 through native resume; configuration/curriculum audit passed.
GPU 0 later became shared with a foreign process. Diagnostics use GPU 1; a separate
Walking throughput benchmark measured 4096/8192/16384 on GPU 3, with 8192 best
at 98k samples/s. A foreign eight-GPU job confounded the 32768 case and stopped
the sweep before StandUp. Long learners retain 4096. Original StandUp sitting
success remains 15/16 at checkpoint 2500; prone/supine remain 0/16. Eight original
checkpoint-2000 CPU/native StandUp videos are checked. See [resource/progress review](reports/official-training-resource-review-2026-09-08.md).
Owned default Walking (3299) and StandUp (2985) are paused for diagnosis; three
Newton learners remain suspended and three earlier SB3 attempts stopped. Walking
3000 has only 0.98% forward response when native pushes are zeroed, despite 36.4%
in its standard preview; it also nearly stops in nominal native and CPU replay.
The official-guided StandUp angular-penalty diagnostic completed 2500: paired
sitting improves from default 10/16 to 15/16, but prone/supine remain 0/16 each.
All eight final CPU/native videos and local packaging are checked; own-policy
behavior reproduction remains open. The user reaffirmed official-recipe-first
reproduction; the prepared second tuning experiment is deferred and was not launched.

Official history audit confirms six historical/current train/play configurations
differ only in Walking logging names. Published policy bytes are traced to official
runtime commits; their exact training run remains unknown. At checkpoint 1500,
both original/owned Walking nearly stand still in CPU replay; neither StandUp
recovers from prone/supine. A replacement observer fixes resumed checkpoint lookup
and advances original task assessments independently while owned runs are paused.
See [source-history evidence](reports/official-source-history-2026-09-08.md).

The corrected CPU/BAM matrix now completes all eight preserved acceptance policies
with finite 61/14 traces and 20 checked videos; all short policies still fail
behavior. Native previews use scoring v2 (`outputs/previews/shared-gpu7-0908-02`).
See [baseline audit](reports/official-baseline-audit-2026-09-08.md) and
[StandUp diagnosis](reports/standup-recovery-diagnosis-2026-09-08.md), and
[Walking condition/transfer diagnosis](reports/walking-transfer-diagnosis-2026-09-08.md).
SB3 campaign learning-rate routing is tested; a four-combination 1e-4 configuration
is prepared but not launched. Default task recipes and native PPO algorithms remain unchanged.

| Component | Current state |
|---|---|
| Source organization | First-party code consolidated under `src/oh_my_duck`; obsolete `training/` packages removed |
| CLI/application | Unified entry point and explicit service composition |
| Core contracts/tool catalog/recording | Tested contracts, tool registry and JSONL event storage |
| Agentic Harness, skills, robot execution | 原生 EDH 会话、真实 Astra 工具与图像、官方 ONNX 推断、CPU MuJoCo/BAM ActionGate 动作、有界命令与传感器停止、`policy_stop` 和独立 Verifier 已运行；固定出生位置的独立 office 导航任务正式 passed，外部障碍接触累计 0，范围为单一 CPU 公寓 seed |
| Perception and policy adapters | Interfaces for future implementation; no runtime capability claim |
| Voice interaction | Qwen ASR、VoiceDesign、Base TTS 与确认后的音色版本已通过 GPU 文件推理；Mac 扬声器、麦克风和 HTTP 服务完成实际音频验证，合成期间停止后无迟到播放；独立 CPU 服务验证 `VoiceSession.speak` 正常播完和播音开始后的停止。Microduck 音频设备和 Harness 接入待完成。见[交互验收](reports/voice-interaction-validation-2026-09-29.md)与[文件推理验收](reports/voice-validation-2026-09-26.md) |
| MuJoCo RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Isaac/Newton RL | Both representative tasks × both native PPO frameworks passed the single-GPU lifecycle and completed replay |
| Newton physics | Actual solver, canonical state/sensors, precise collisions, BAM cadence, DR and penalties audited |
| Published StandUp reference | Official frozen policy passes 64/64 CPU reset samples and 4/4 cases in each native backend; 12 videos checked; own training reproduction remains open |
| Task behavior | All five-iteration policies fail behavior gates; long training and convergence acceptance remain |
| Task replay | All 24 replay contexts completed; 60 videos and finite 61/14 traces checked |
| CPU rehearsal/sim2sim/local packages | All eight corrected CPU/BAM executions revalidated; 20 videos checked; learned behavior remains unverified |
| Multi-GPU | Native RSL MuJoCo DDP previously passed; Newton DDP acceptance pending; old job `a52ff51b` no longer exists in the platform API; no distributed SB3 gradient claim |
| Hardware | Unavailable; all hardware acceptance deferred |

The consolidated batch at `5971dca` passed training and resume for all eight
combinations. Four SB3 exports and local packages passed there. After the merge,
all four RSL checkpoints passed export, numerical parity, finite scalar/penalty
checks and local schema-2 packaging using source `6303ca4`. Evidence:
`outputs/postmerge-rsl-export-0906-01/result.json`. The earlier failures remain
preserved; ordinary parity uses elementwise tolerance and extreme stress inputs
use a reported per-action-vector infinity norm tolerance.

The complete replay matrix used `a0b1f0f`; all eight policies completed both
simulation backends and CPU/BAM rehearsal without runtime errors. Behavior failed
as expected for short smoke checkpoints. Explicit MuJoCo OSMesa video and native
Newton video produced 60 verified 720p clips. SB3 curriculum restoration was then
validated at `b65e8f9` on both tasks/backends, including a second saved-state resume.

Latest tests passed 27 lightweight and 72 task/SB3 cases. Earlier installed-runtime
checks passed seven Isaac and two Newton binding tests. Wheel resources/licenses,
bytecode exclusion, three environment locks and local Markdown links were checked.
See [complete acceptance evidence](reports/rl-pipeline-acceptance.md).

Single-GPU work runs directly on the development host. Following the scheduled
campaign failure, the user also authorized the eight-GPU campaign on the local
H200 host on 2026-09-08; multi-GPU scheduler submission remains available. W&B remains offline. Current host workloads contend for GPUs,
so observed throughput is not an isolated hardware benchmark.

See [domain refactor evidence](reports/domain-refactor.md),
[RL acceptance](rl-reproduction.md), [architecture](architecture.md), and
[historical implementation log](reports/implementation-history.md).

See [headless rendering evidence](reports/rendering-validation.md) for the explicit software renderer and preserved failed attempts.


## Task organization and full-training preparation — 2026-09-07

Task recipes now live in 14 family directories with separate `environment.py` and
`ppo.py`; flat/rough/backlash variants keep the existing registry. All 33 compiled
task catalog entries match the prior semantic inventory (module paths excluded).
Thirty lightweight and 74 task/SB3 tests passed, including native periodic
checkpoint reload and GPU/rank isolation. Full campaigns have a 64-env smoke,
resume and 4096-env capacity gate before the official full iteration budgets.
Long-training submission and worker gates are recorded separately below.


The new campaign worker completed two reduced-size end-to-end gates (RSL-RL and
SB3 Walking) in `outputs/campaign-gate-0907-01`. Every stage completed, including
periodic SB3 bundle resume, final package and both-backend/CPU video. Both reported
`behavior_failed`, with no execution error. These orchestration gates used 64
environments and a two-iteration final segment; the real 4096-environment capacity
gate runs on the allocated training node before long training. Newton binding
checks also passed (two tests). Source/config provenance now includes nested
experiment manifests. Full submission follows from the committed snapshot.


The scheduled attempt `omd-rl-full-0907-01` (platform ID `b2bab260`, original
queue ID `local-662cf40c7dfb`) is **Failed**. Its log is empty, the platform exposes
no log pods, and no campaign manifest exists; the root cause remains unknown.
A separate local attempt `full-local-0908-01` started on eight H200 GPUs using
the same immutable source `1f45996` and training budgets. All eight smoke stages
completed; subsequent gates and full training remain in progress. Failed scheduler
evidence is preserved; local launch metadata is in
`outputs/jobs/omd-rl-local-0908-01/launch.json`.
See [the allocation and evidence record](reports/full-training-2026-09-07.md).


At 2026-09-08 02:08 UTC all eight local runs had entered full training. Early
returns improved relative to initial records; SB3 StandUp shows substantial
regression from its early peak. No learned behavior acceptance is claimed.
See [the measured reward snapshot](reports/reward-trends-2026-09-08.md).


## Triage and one-GPU continuation — 2026-09-08

The two regressing MuJoCo SB3 runs are stopped with artifacts preserved. Six
retained runs resumed native checkpoints on GPU 7 at source `b0fa5fe`; the other
GPUs are released from this campaign. Checkpoint video previews are available at
`outputs/previews/shared-gpu7-0908-01/index.html`. Early RSL task videos show
partial skill progress, with no complete behavior acceptance. A controlled native
SB3 update comparison demonstrates substantially lower KL with smaller learning
rates; stable long-training convergence is still unvalidated. See the
[recovery, diagnosis and video report](reports/rl-recovery-2026-09-08.md).


At the 2026-09-08 04:04 UTC status review, Newton SB3 StandUp also showed
return decline (last 50 logged records 11.95 versus 14.45 previously), elevated
KL (recent values 0.13–0.29) and failure of all four spawn checks on its preceding
preview. It was paused for diagnosis under the user’s standing instruction. Its
latest complete checkpoint at cumulative iteration 1500 is preserved with an
operator-stop record. Five runs continue on GPU 7; this is a triage decision,
not proof that the stopped configuration could never converge.


Official MuJoCo/RSL learned-behavior reproduction remains **unverified**. The
2026-09-08 static audit aligns task/PPO configuration, 228/229 MDP definitions
modulo local imports, BAM and core dependency pins, and the subsequent real-runtime audit verifies the Entity-based StandUp reset
state writes in 15 cases. Matched original/refactor learned-behavior comparison
remains outstanding. See [baseline audit](reports/official-baseline-audit-2026-09-08.md).


Current goal: reproduce official MuJoCo/native RSL learned behavior first, then
match it in Newton and SB3. Isolated pinned-original StandUp PPO smoke and official
export/scalar audit passed; both implementations passed 64-env/5-iteration smoke.
Compiled-model equality, exact reset state/RNG parity and first-reset actor
observation equality are verified. Repeated original runs also show contact-force
and trajectory nondeterminism; long-run behavior is still unverified. Walking
controls and the complete sensor-refresh comparison are in progress. See the
[baseline audit](reports/official-baseline-audit-2026-09-08.md) for evidence and limits.


Walking runtime controls now also match compiled arrays and initial/subset actor
and critic observations; all four original/owned representative PPO smoke runs
completed. A published-policy rehearsal exposed a Walking scoring false positive:
standing almost still met the global RMSE threshold. Scoring v2 now requires signed
motion response in every commanded segment; the preserved published-policy trace
is reclassified `behavior_failed`. This changes acceptance scoring, not training
semantics. Six focused protocol/reset tests pass. See the baseline audit for the
measured commands, responses and remaining interpretation limits.


Official-first resource priority is now active: two owned MuJoCo learners and one
isolated original StandUp control are training on GPU 7. Three live Newton
learners are suspended with SIGSTOP, retaining process/checkpoint state for later
SIGCONT; they are separate from the three earlier stopped SB3 attempts. The
original control passed 4096-env capacity/export and normalized numerical parity
before this status update and is running its 15,000-iteration budget. Baseline
iteration time improved by about 1.5× after suspending extensions. No long-run
behavior success is claimed. Evidence and safe resumption identity records are
linked in the [baseline audit](reports/official-baseline-audit-2026-09-08.md).


CPU/BAM behavioral acceptance requires revalidation: the pinned CPU controller
matched DOF-friction constraints by joint ids, unlike the training Warp path.
The owned CPU adapter now uses actual DOFs while retaining native motor/sag and
friction formulas. Three real-physics/reset tests pass, including independent
Jacobian-force projection and unchanged motor torque. Old traces remain preserved;
corrected policy replay has since completed without restoring the missing behavior (see current baseline audit). This is an evaluation/CPU-load correction,
not evidence that training converged. See the baseline audit.


## Current video review

`outputs/previews/review-0908-01/index.html` links six existing clips with explicit
labels: default-recipe owned Walking/StandUp, the same Walking policy's CPU
rehearsal, published StandUp in MuJoCo/Newton, and the separate angular-penalty
diagnostic. The gallery does not conflate published-policy replay with own
training success. All six local video targets were checked.
