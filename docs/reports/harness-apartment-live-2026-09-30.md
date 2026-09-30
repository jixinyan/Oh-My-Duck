# EDH 原生 MicroDuck 公寓闭环验收（2026-09-30）

真实 Astra 会话 `d1213739-33a5-4c4b-99dd-4e48d79f03e9` 的任务 `d5dedd29-ef60-4c7d-9358-e21abecccf4c` 使用固定 office 目标、seed `20260929`、初始位置 `(0,0,0)`。上层模型是 `gpt-6-astra`，连接已有 OpenAI-compatible 服务。运行使用固定 EDH `8a5e685b22d032207f53db20454f0992a4ad60fd`，该源码工作目录的 Git 管理内容保持清洁；OMD 基础提交为 `dc19b03e261c187810560a177eb2b44ca25aeae8`，测试时保留工作目录中的用户草稿。

Planner 实际读取 `head_rgb`、ToF、IMU、关节和里程计，调用 `scene_info` 与官方策略目录，从 velstand 切换到 alpha_walking，设置前进命令 `[0.4,0,0]`，并读取物理位置。velstand 确认暂停时控制序号为 185、世界 x 约 `0.00213 m`；alpha_walking 的第 244 个控制步后 x 约 `0.18279 m`。本任务共完成 550 个经 ActionGate 接纳的 14 关节控制步，对应 2200 个 MuJoCo/BAM 子步；停止时位置为 `(0.57675565,0.06835097,0.11706653) m`。前进命令与位置变化均来自原始事件，尚不构成 Walking 速度跟踪通过。

事件 822 调用 `microduck.finish_policy`，事件 824 确认操作，事件 827 发布原生 `policy_stop` 结束边界。独立 Verifier 于事件 839 通过 `verification.check` 读取新鲜目标检查，于事件 850 提交正式 `verification.completed`，结果 `status=failed`、`goal_reached=false`。原生证据记录 office 房间范围 `x=[0.7,3.8] m`、`y=[-0.8,0.8] m`、额外 clearance `0.2 m`，机器人位置仍在目标外，连续达标控制步为 `0/5`；机器人当时满足直立姿态阈值。Planner 于事件 883 使用原生 `tasks.abandon` 结束该任务，最终 run 状态为 `failed`。

同一真实会话的第二项任务 `8e00607b-0a0b-46ea-8e30-8885c7666fcf` 获得新的 run、execution 和 lease 身份，沿用物理 episode `77a919291f8a4421b46329febe25ee70`，从序号 550、位置 `(0.57674867,0.06833746,0.11706470) m` 开始，空间位置仍在 office 目标外。Planner 读取头部 RGB、里程计与执行状态，沿用此前已确认选择的官方 `alpha_walking`，再次设置非零前进命令，并检查位置变化。该项任务新增 275 个经 ActionGate 接纳的控制步和 1100 个 MuJoCo/BAM 子步；序号 699 的位置为 `(0.71212481,0.54382909,0.12336411) m`，序号 825 为 `(0.74672850,0.68282476,0.11616119) m`。这项任务有自己的物理运动与正式检查。

第二项任务事件 432 发布原生 `policy_stop` 结束边界；事件 454 的独立 `verification.checked` 读取序号 825 的原生位置 `(0.74673419,0.68282937,0.11614396) m`、直立姿态、20 个原生接触，以及连续达标控制步 `131/5`，目标条件 `goal_reached=true`。事件 461 的正式 `verification.completed` 为 `status=passed`，verdict ID 为 `a068e7ba-d099-4425-95d6-d9c939cc1909`；Planner 于事件 502 使用原生 `tasks.finish`，事件 504 为 `run.succeeded`，引用同一 verdict。该次 office 导航在固定 GT 下通过；单次成功不建立速度跟踪或重复成功率结论。

独立第三视角会话 `e929695e-8300-41a0-b85c-f215c576d603`、任务 `6b8b5c45-2f8a-45fa-8d28-e95a7465c668` 从相同 seed 的走廊初始位置开始。Astra 读取策略目录、公开房间信息、头部 RGB、ToF、IMU、关节与里程计，先执行 velstand，再在确认停止边界选择 alpha_walking；已完成的命令包括前进 `[0.4,0,0]`、横向及转向调整、零速度和最后的 velstand。机器人由序号 187 的 `(0.00212,-0.00007,0.11667) m` 移动至序号 815 的 `(0.58439,0.02286,0.11688) m`，之后在目标外附近持续运动。该任务共执行 4000 个经 ActionGate 接纳的控制步和 16000 个 MuJoCo/BAM 子步；Gate 于事件 5145 以 `budget_exhausted` 确认结束。事件 5157 的新鲜原生目标检查为 `goal_reached=false`，位置 `(0.58169,0.02177,0.11674) m`、连续达标 `0/5`；事件 5164 的正式 Verifier 状态为 `failed`，事件 5196 为 `run.abandoned`。本任务没有调用 `finish_policy`，其视频显示真实预算终止和失败判定。

首项任务原生导出位于 `.cache/demo/d5dedd29-ef60-4c7d-9358-e21abecccf4c/`，包含 900 条完整事件、110 张按每五个控制步记录的 `observer_follow.png` 帧、头部 RGB 与原生视频。第二项任务导出位于 `.cache/demo/8e00607b-0a0b-46ea-8e30-8885c7666fcf/`，包含 521 条完整事件、55 张第三视角帧与原生视频。第三视角帧各为 `640×480`，有独立原生 `observation_id`、物理序号、仿真时间和 SHA256。`scripts/accept_harness_replay.py` 对两项任务的原始工具记录、ActionGate/Verifier 身份、正式结果、每张第三视角 PNG 的 SHA256 和单调仿真时间进行了重新核验；第二项额外检查同一会话、物理 episode 与控制序号连续性、此前确认的策略选择及本任务的非零命令和位置变化。EDH 控制台浏览器 DOM 显示真实工具记录、验证状态，头部图像解码尺寸 `320×240`，第三视角图像解码尺寸 `640×480`。

独立第三视角任务的导出位于 `.cache/demo/6b8b5c45-2f8a-45fa-8d28-e95a7465c668/`，包含 5223 条原生事件、800 张 `observer_follow.png` 帧、头部图像及原生视频。operator 相机距离为 `0.5 m`、俯角为 `-45°`、方位角为 `45°`；真实 100 控制步测试通过图像分割核查，项目审查者核对了该任务序号 815 的画面，机器人身体完整可见。原始导出复核了 800 张相机 PNG 的 SHA256、仿真时间、真实工具完成、策略切换、非零命令、位置变化、Gate 停止边界和正式失败判定。

上述两项连续任务启动时的运行模块 SHA256：`integrations/edh/server.mjs` 为 `ff8bdcd907a2449af21723b67bd9bcd7d1f7021651c95b99b3860cee537d09bb`；`src/oh_my_duck/integrations/edh_native.py` 为 `622c98477803ccc5619c87cf7eea868527f95730accb6134f420ecae5b092dbf`；`src/oh_my_duck/robotics/backends/simulation.py` 为 `672670c923166096896f5c3739ac0188c3648dd3050b6be72e71ba3235db364c`；`src/oh_my_duck/robotics/microduck/official_policies.py` 为 `576d5b0372257a5e5f0f83b5bc52d3dd63787d1ca43e08a7b26f3efbabe44dd7`。官方策略 revision 为 `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`，实际 alpha_walking ONNX SHA256 为 `e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c`。

独立第三视角任务启动时的 SHA256：`server.mjs` 为 `ff8bdcd907a2449af21723b67bd9bcd7d1f7021651c95b99b3860cee537d09bb`；`edh_native.py` 为 `beec5ce41fae2c3ab411f905be5a7a7e26915d12540dd6a6f5558ed9c5e9991e`；`simulation.py` 为 `61085ad9cf2a27598c03324362c464e959058a65a3c05963ae8004a87011d92a`；`official_policies.py` 为 `576d5b0372257a5e5f0f83b5bc52d3dd63787d1ca43e08a7b26f3efbabe44dd7`。EDH 固定源码提交为 `8a5e685b22d032207f53db20454f0992a4ad60fd`，Git 管理内容保持清洁。官方模型文件 SHA256 与上述连续任务相同。

独立取消任务 `88c08463-0d80-4dff-bf08-fbb4661e552a` 的原生导出保存于 `.cache/demo/88c08463-0d80-4dff-bf08-fbb4661e552a/`，共有 352 条事件。该任务先完成 205 个 ActionGate 控制步及 820 个物理子步；事件 299 为设备确认的暂停边界，动作计数保持 `205/820`，事件 351 为 `run.cancelled`。确认边界以后没有新增 `execution.updated` 或 `simulation.frame`。这项取消记录来自独立较早会话，其源码版本与上述第三视角任务不同；复核范围为该次原生记录中的取消后动作计数。

当前源码还通过独立真实 worker 的并发任务边界测试：加载 CPU MuJoCo/BAM 公寓和官方 velstand ONNX，执行 9 个原生 ActionGate 控制步后确认暂停。在真实第三视角渲染占用唯一物理 owner 时，第一项任务提交 `select_policy` 请求；任务关闭后，该排队请求收到 active-lease 拒绝。第二项任务以相同方式提交正在等待 owner 的 `set_command`；任务关闭后，该请求也收到 active-lease 拒绝。另以第一项任务身份发送旧 `set_command`，得到同样拒绝。第三项任务保留同一 episode、控制序号 9 和未变化的 `command_block`。该测试使用真实 worker、官方 ONNX 与物理 owner；范围为暂停及关闭时的控制请求身份，没有把 9 个站姿控制步作为行走行为验收。

复现依赖与启动命令见[原生 EDH 公寓部署](../harness-native-integration.md)。完整原始导出后执行：

```sh
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py .cache/demo/d5dedd29-ef60-4c7d-9358-e21abecccf4c --expected-verdict failed --expected-stop-reason policy_stop
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py .cache/demo/8e00607b-0a0b-46ea-8e30-8885c7666fcf --prior-export .cache/demo/d5dedd29-ef60-4c7d-9358-e21abecccf4c --expected-verdict passed --expected-stop-reason policy_stop
TMPDIR="$PWD/.cache/tmp" PYTHONPATH=src .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_replay.py .cache/demo/6b8b5c45-2f8a-45fa-8d28-e95a7465c668 --expected-verdict failed --expected-stop-reason budget_exhausted
TMPDIR="$PWD/.cache/tmp" .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_cancel.py .cache/demo/88c08463-0d80-4dff-bf08-fbb4661e552a
TMPDIR="$PWD/.cache/tmp" PYTHONPATH="src:$PWD/.cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/harness/physical-runtime/src" .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_lease.py
```

原生 EDH 当前执行 perpetual 策略；episodic 策略可在目录中查看，尚未在该接入中完成执行验收。真实硬件、Walking 命令速度跟踪与导航重复成功率仍需各自验收。
