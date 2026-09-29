# Walking 低速干预完整训练启动记录（客户端 2026-09-29）

服务器实验目录沿用 2026-09-30 标识。`jd_B300` 的正式源码为独立的 `b8c36b29b847f117043c961ac9ecb6a0b9b3deec` worktree，路径为 `.job-sources/b8c36b2/`，已跟踪文件保持干净。该源码保留先前通过门禁的原始训练配置；`walking-low-speed-boost.json` 的 SHA-256 为 `9f6ee7639ca21ea489a37c1e2629f88a281247116ff75515c4a0ccca5239fe73`。每组使用 8192 环境、50000 次更新和 seed 42，RSL-RL 与 SB3 各自使用原生 PPO。配置包含每 1000 次更新的周期预览、训练后导出与本地封装、标准双后端 sim2sim、CPU/BAM 演练和双后端无推扰评估；W&B 模式为 `online`。

八组训练前产物均经现有 `validate_preparation` 检查，完整 run spec、六个必需阶段、两份导出文件以及固定源码均匹配。四组 MuJoCo 采用 `walking-low-speed-boost-20260930-02` 内的 `prepared` 结果；该目录中的四组 Newton 初次尝试失败，仍保留原始记录。四组 Newton 采用 `walking-low-speed-boost-20260930-03-newton-gates` 内完成的 `prepared` 结果。八份门禁文件的 SHA-256 和来源路径保存在新实验的 `campaign.json`，验签结果中的 `critical_input_diff` 均为空。

新实验输出为 `outputs/experiments/walking-low-speed-boost-full-20260930-01/`，每组以原有 worker 的 `--prepared` 参数启动，并在自己的 GPU 上执行完整训练与最终评估。监督程序保存在项目忽略目录 `.cache/rl-walking-full-20260929/`，记录来源、进程、资源等待和完成状态。所有 worker 的 `TMPDIR` 指向项目的 `.cache/tmp`。

当前六组正式训练已产生原生 PPO 更新：GPU2/3 分别执行 MuJoCo SB3 control/boost，GPU4/5 分别执行 Newton RSL control/boost，GPU6/7 分别执行 Newton SB3 control/boost。GPU0/1 的 MuJoCo RSL 两组等待其他项目释放设备；启动时占用进程为 OpenSO101 PID 485398、485390。语音服务完成实际验收并退出后，GPU5 使用率为 1 MiB、0%；经明确授权，首次实验目录写入 `release-gpu5.json`，包含 `release_gpu5=true`、上述完整 `source_commit` 和该实验绝对路径 `campaign`。监督程序核查 GPU5 空闲后启动 Newton RSL boost。当前没有策略行为验收结果。

## 产物身份与运行目录

固定源码 worktree 的 `artifacts` 指向服务器共享资产目录。Newton `walk` 资产 fingerprint 为 `cbe7868687f4a0fbf51c62eb3032bf02d05d2e650dc211692d4e6535372dca1d`；原生 `require_asset("walk")` 已验证对应 `build.json`、文件清单和哈希。环境代码检查 `SolverMuJoCo`，固定源码的已跟踪文件保持干净。

GPU4/6/7 的 Newton 首次 attempt 保存在 `outputs/experiments/walking-low-speed-boost-full-20260930-01/`，三份 `result.json` 均为 `execution_failed`，错误为资产目录不可见。当前三组运行于独立目录 `outputs/experiments/walking-low-speed-boost-full-20260930-02-newton-retry/`，PID 分别为 495834、495835、495836；新目录的 `campaign.json` 分别记录首次失败和当前运行的两个 attempt。后台汇总文件 `outputs/experiments/walking-low-speed-boost-full-20260930-01/attempts-summary.json` 保存全部八组的每次运行路径与状态。GPU2 后续出现其他项目 OpenSO101 进程 PID 493789；本项目保留该资源记录。

## 启动核查

MuJoCo SB3 的 control 与 boost 组已在各自 `full.log` 连续记录真实 PPO 更新，W&B 在线 run 分别为 `46afwkq7`、`4fmav7qs`。一次性资源核查时，GPU2 总显存 275040 MiB、已使用 16900 MiB；本项目进程约 11978 MiB，后来进入的 OpenSO101 进程约 4853 MiB。本项目训练仍继续更新。Newton RSL control 已记录 `Learning iteration 0/50000`，W&B 在线 run 为 `anxyia9j`；GPU5 的 Newton RSL boost 已连续记录 `Learning iteration 0/50000` 至 `10/50000`，W&B 在线 run 为 `0mpdlqb7`。Newton SB3 control 与 boost 均记录至少 196608 个环境步，W&B 在线 run 分别为 `vc9mq86q`、`gg7jfzdl`。上述六组均已产生真实 PPO 更新；当前没有策略行为验收结果。本实验的行为比较依据最终结果，受其他项目资源使用影响的运行时间不用于性能结论。

GPU5 的 Newton RSL boost 组已有 capacity 记录，显示 8192 环境、5 次更新以及导出阶段完成；该组的 `result.json`、`capacity.log` 和原生 `run.json` 没有保存 GPU 显存峰值。本轮在语音服务退出、GPU5 明确释放后启动该组 learner。

监督程序继续保存设备等待、进程身份与最终结果，并按固定步数生成策略预览；当前没有执行中间行为评估。
