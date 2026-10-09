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
- policy 工具调用权限来自 `tools_and_limits.allowed_tools`；所选 policy
  产生的关节 action 通过 ActionGate 接纳。
- `omd harness --max-output-tokens` 设置 256–8192 的模型输出预算，
  默认 4096；原生 context compaction 使用对应的输出空间。

实现位置为 `integrations/native_client.py`、`cli/voice_task.py`、
`voice/session.py`、`integrations/edh/session.py` 和
`integrations/edh/roles/planner.md`，均位于 `src/oh_my_duck/`。
独立检查通过 `omd validate task-continuation` 和
`omd validate voice-context` 提供，源码位于 `validation/harness/`。

## CPU 物理检查

`task-goal-continuation-20261009-02` 使用源码
`e8caab2903d930296e9ed88c48c722fa237da6c2`、固定原生 Harness
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
这个检查的初始位置保持在目标之外，其范围为任务绑定与执行生命周期。

全部检查共执行 300 次控制、1,200 次物理步骤。
远程原始目录为
`/home/jixin/workspace/code/Oh-My-Duck/outputs/acceptance/task-goal-continuation-20261009-02/`，
保存两份 action 记录、`native-tasks.json` 与 `result.json`。
本地副本位于
`outputs/acceptance/voice-task-context-20261009-06-install/`。
保存记录的独立复核确认四项不同的目标记录、两个原生执行、
300 次控制、1,200 次物理步骤和任务间连续状态，
结果位于同一目录的 `independent-task-goal-audit.json`。

## 接口与安装检查

源码 `3c5179903f5b7c16c5bd653d067231debc5c756f` 的四项接口测试文件
通过 101 项测试及 5 项子测试，包含历史引用格式、等待条件、共享应用接口
与场景接纳检查。
独立安装的 wheel 和 sdist 检查通过 438 个源码与资源文件、三份 license、
29 项 CLI 调用，并通过已保存真实控制记录的 ONNX 与物理指标复核。
这些保存记录使用其各自注明的执行版本。
安装报告位于 `outputs/acceptance/voice-task-context-20261009-07-install/`。

## 模型任务验收范围

两项连续录音任务的正式验收正在执行。完成条件包含实际 ASR 指令、
原生历史引用、Planner brief、经验搜索、当前物理状态、独立目标保持、
独立 Verifier、已确认音色反馈和全部进程退出。
保存结果通过 `omd validate voice-context` 复核。
CPU 任务检查与安装检查分别保留对应的范围。

GPU 验收与 RL 保持停止。当前 Newton 运行、语音交互质量、
广泛场景表现和 Microduck 设备需要对应的执行证据。
