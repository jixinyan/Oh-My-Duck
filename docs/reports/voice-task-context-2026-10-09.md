# 连续语音任务与原生历史引用

每项录音指令通过 Qwen ASR 转为文字，再由原生 Harness 接纳和执行。
`context_run_ids` 明确选择同一打开会话中的已结束任务，最多四项。
原生服务核查任务归属和结束状态，将历史指令、结果、正式 verdict 与 skill
引用写入新任务的 `history_summary`。Planner 通过 `skills.search` 和
`skills.load` 读取持久经验，当前传感器提供当前位置和物理状态。

## 当前实现

- `NativeTaskClient` 与 `HarnessBridge` 提交明确的历史任务编号；
  接纳之前检查数组、数量、重复编号和 ASCII 格式。
- `voice-session execute_recording` 在 ASR 请求之前检查历史引用；
  `voice-task --audio FILE FILE --context-previous-task` 在同一会话中依次
  执行录音，并让后续任务引用上一项成功任务。
- 新任务等待原生 session 完成收尾并返回 `ready`，关闭会话要求
  `closed` 与 `resources=released`。
- 每项任务保存实际 `task_start`，包含 episode、sequence、时间、位姿、
  速度、policy、关节和初始目标检查。任务绑定保持这些物理状态。
- `goal_scope_id` 与 `goal_bound_sequence` 标识当前任务的目标证据；
  `held_ticks` 从零开始，新的实际 policy 控制步增加计数。
  目标范围、upright 条件和原始任务预算继续由场景管理。
- 原生 `open_task` 初始化目标记录；`bind_task` 核查任务身份。
  `execution.start` 保持当前任务的目标编号与起始 sequence。
- policy 工具调用权限来自 `tools_and_limits.allowed_tools`；所选 policy
  产生的关节 action 通过 ActionGate 接纳。
- `omd harness --max-output-tokens` 设置 256–8192 的模型输出预算，
  默认 4096；原生 context compaction 使用对应的输出空间。
- 原生 `purpose=compaction` 标识摘要辅助调用。模型适配代码说明这项
  请求的作用范围，摘要保留原有用户指令、当前意图、待执行动作和验证要求。
  工具调用与 agent loop 继续由原生 Harness 管理。
- 暂停证据保留原始物理记录与执行身份；policy 切换后的进度检查同时核查
  选择回执、当前 policy 和停止计数。连续任务的初始指令以 `task_start.sequence`
  为起点，保持原有的 75 次控制、300 次物理步骤与已确认边界。
- 保存记录的检查读取原生 `edh.assignment-history.v1`，核对 Planner
  与公开任务记录的身份、状态、模型、工具和最近观察，再核查历史摘要。

实现位置为 `integrations/native_client.py`、`cli/voice_task.py`、
`voice/session.py`、`integrations/edh/session.py` 和
`integrations/edh/roles/planner.md`，均位于 `src/oh_my_duck/`。
独立检查通过 `omd validate task-continuation` 和
`omd validate voice-context` 提供，源码位于 `validation/harness/`。
Node 模型适配代码位于仓库根目录的 `integrations/edh/model-context.mjs`，
传输继续使用原生 OpenAI Responses 或 Chat Completions adapter。

## CPU 物理检查

`task-goal-continuation-20261009-03` 使用源码
`8d6cac45aeb7702fafc182b3254becab7b999e4d`、固定原生 Harness
`8a5e685b22d032207f53db20454f0992a4ad60fd` 和官方 policy
`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。
执行环境为 MuJoCo 3.8.0、BAM 1.0.1、ONNX Runtime 1.29.0、NumPy 2.5.2，
renderer 为 Mesa llvmpipe，CUDA 保持未初始化。

直接环境检查执行两项连续目标记录，每项 75 次控制、300 次物理步骤。
任务绑定保留 `qpos`、`qvel`、`ctrl`、上一 policy action、episode、sequence
与时间。每项任务的十次只读检查保持 `held_ticks=0`；第五次实际控制之后
可以满足原有的五次目标保持要求。完整执行保留 upright 与实际停止条件。
外来任务编号和未完成的 policy inference 在接纳位置被拒绝。

原生 `MicroDuckWorkerSession` 另外执行两项连续任务，每项 75 次控制、
300 次物理步骤。后一任务的 `task_start` 与前一任务最终位姿、速度、
episode 和 sequence 一致，取得新的目标记录。两项执行通过 ActionGate
结束，记录 `policy_stop`、device confirmation 和资源释放。
每项任务从 `task_start`、`execution.start` 到结束保留相同的目标编号
和起始 sequence。
这个检查的初始位置保持在目标之外，其范围为任务绑定与执行生命周期。

全部检查共执行 300 次控制、1,200 次物理步骤。
远程原始目录为
`/home/jixin/workspace/code/Oh-My-Duck/outputs/acceptance/task-goal-continuation-20261009-03/`，
保存两份 action 记录、`native-tasks.json` 与 `result.json`。
本地副本位于
`outputs/acceptance/voice-task-context-20261009-09-install/`。
保存记录的独立复核确认四项不同的目标记录、两个原生执行、
300 次控制、1,200 次物理步骤和任务间连续状态，
结果位于同一目录的 `independent-task-goal-audit.json`。

## 接口与安装检查

源码 `8d6cac45aeb7702fafc182b3254becab7b999e4d` 的四项接口测试文件
通过 101 项测试及 5 项子测试，包含历史引用格式、等待条件、共享应用接口
与场景接纳检查。
源码 `9a9ea30dc576c1212429096eb5b3ade16da1af01` 的独立安装 wheel 和 sdist
在 macOS 和 Linux 均通过 438 个源码与资源文件、三份 license 检查。
macOS 通过 29 项 CLI 调用，包含已保存真实控制记录的 ONNX 与物理指标复核；
Linux 通过 27 项 CLI 调用，以及实际连续语音和导航记录复核。
Mac 安装报告位于 `outputs/acceptance/voice-context-evidence-20261009-02-install/`，
Linux 安装报告位于 `outputs/acceptance/voice-context-independent-20261009-01/`。
保存控制记录保留其各自的执行版本。

## 模型任务验收范围

两项连续录音任务均取得原生 `succeeded` 和独立 Verifier `passed`。
运行源码为 `8d6cac45aeb7702fafc182b3254becab7b999e4d`，独立安装的检查源码
为 `9a9ea30dc576c1212429096eb5b3ade16da1af01`。这两个版本的机器人、
原生 Harness 接入、语音和控制源码内容相同。
模型使用 `gpt-6-luna`、`high` 和 8192 输出 token；仿真为 CPU MuJoCo/BAM。
ASR 使用 Qwen3-ASR-0.6B revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，
反馈使用 Qwen3-TTS-12Hz-0.6B-Base revision `5d83992436eae1d760afd27aff78a71d676296fc`。

| 项目 | 语音导航 | 后续观察与停止 |
| --- | --- | --- |
| run ID | `7a0afece-b0c4-4d92-bae5-d8f245c600d9` | `c1a53951-7cdb-4958-aae3-5c8c73f37f7c` |
| 正式 verdict | `276fad80-7504-470b-92c3-8f7e20b0874c` | `140d0f4e-b35c-474d-92cb-3b2ae09adedd` |
| 控制次数 | 670 | 75 |
| 物理步骤 | 2,680 | 300 |
| 原始事件 | 1,467 | 317 |
| 原始 PNG | 149 | 17 |
| 最终停止计数 | 55 | 130 |
| 当前任务目标保持计数 | 98 | 75 |

第一项任务接收录音指令，模型调用四次 `microduck.walk`、两次
`microduck.rotate`，完成办公室导航与传感器读取。后续指令经 Qwen TTS
生成后交给 ASR，明确引用第一项任务的结果；Planner brief 保留相应
run ID、原始指令、正式 verdict，实际调用两次 `skills.search`。
经验搜索结果和原始事件保留在导出记录中。

两项任务使用同一 session `98aa6070-6b75-414c-b503-fdb20470e6b9` 和物理
episode `1ab280f0f8e0418cb89faafec262cf64`，sequence 从 0 增加到 670、745。
后一任务保留前一任务的最终位姿与速度，使用新的目标编号，并从
`held_ticks=0` 开始计数；两次执行均由 ActionGate 确认 `policy_stop`。
原始目标要求、upright 条件、4,000 次控制与 1,800 秒任务预算保持原有内容。

`omd validate voice-context` 通过 1,784 项原始事件、166 张 PNG 的完整
文件检查与两个 PCM16、24 kHz、单声道反馈 WAV 的 SHA256 和数值检查。
persona、voice ID、revision、声音资料和数据库保持一致；会话确认
`closed` 与 `resources=released`。
第一项导航另外通过 `omd validate replay --require-metric-tools
--require-stop-progress`，包括 23 项停止进度检查与 134 张 observer 图片。

跨会话历史引用的实际 API 检查返回 400，任务创建次数为零，所属会话
资源释放。暂停检查拒绝 11 类不一致记录，包含 policy 回执、当前状态、
停止计数、原始暂停、控制计数与物理步骤。原始记录保持原有内容。
五个所属服务或命令进程确认退出，原有服务端口已关闭。

原始任务资料位于远程
`/home/jixin/workspace/code/Oh-My-Duck/outputs/acceptance/cpu-voice-context-20261009-05/`。
独立复核资料位于 `outputs/acceptance/voice-context-independent-20261009-01/`，
包含 `voice-context.json`、`navigation-replay.json`、`guard-rejection.json`
和 `package-audit.json`。跨会话检查位于
`outputs/acceptance/native-context-boundary-20261009-01/`。

GPU 验收与 RL 保持停止。当前 Newton 运行、语音交互质量、
广泛场景表现和 Microduck 设备需要对应的执行证据。
