# 原生模型任务中的运动等待与同步进度

日期：2026-10-13。

## 实现与来源

`microduck.wait_for_motion(max_wait_s)` 使用 1–90 秒窗口，默认 90 秒。
当前控制步完成原生发布与 controller 更新之后，工具读取同一 sequence 的物理位置、
目标测量、停止样本及 `goal_check`。等待结束后的同步读取包含控制步和通信耗时。
取消等待调用保留当前 policy Task；停止动作使用 Harness 原生停止流程。

| 内容 | 固定来源 |
| --- | --- |
| 实际 CPU 动作与完整模型任务 | `7fdaf000d1851e8c981d8cb46a06418d4fb379a9` |
| 安装包与公开等待记录检查 | `4fe8dd88d1da91af1962ff88117fd39b14ebb255` |
| EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy catalogue | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| YOLO26 权重 SHA256 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

实际任务和安装检查使用独立、没有 tracked 改动的源码目录。
安装版中的原生 session、Node 工具与 planner 指导和任务运行版本逐字节一致。
场景是官方 8 × 6 米 apartment，CPU MuJoCo/BAM、OSMesa、llvmpipe，
`CUDA_VISIBLE_DEVICES` 为空。官方观测、动作、BAM 和控制周期保持原有定义。

## 实际 CPU 动作检查

| 项目 | 实际结果 |
| --- | --- |
| 控制 / 物理子步 | 461 / 1,844 |
| 使用策略 | `velstand`、`alpha_walking` |
| 独立 ONNX 动作重算 | 461 项，最大误差 0.0 |
| 原生序列化更新 | 467 项 |
| 等待结果 / observer PNG | 11 / 92 |
| 一秒窗口中的 sequence | 2、4、77 |
| 九十秒窗口中的 sequence | 218、362 |
| 确认边界重复读取 | 75、411、461；物理计数和测量保持一致 |
| 非法参数拒绝 | 七项 |
| 取消等待调用 | 当前动作继续执行 |
| 目标接近误差 / 容许误差 | 0.0240209885 / 0.05 米 |
| 最终停止样本 | 55，要求五个 |
| 停止原因 | `policy_stop` |
| 退出与资源 | 所属进程退出 0，控制端口关闭，会话释放资源 |

每条等待结果检查真实位置、controller sequence、停止样本与原生执行状态。
原始动作、测量、图像与工具结果使用各自 SHA256 核验。

## 完整 Luna 模型任务

模型使用 `gpt-6-luna`、Responses、`high`、8192 output tokens，保存 61 次实际
`response.completed`。Agent 从原始 YOLO26 检测中选择有效小型目标，
在不同物理 sequence 查询同一目标，接近测得的表面点，并保持停止。
距离来源是仿真 RGBD；原始检测 label 为 `orange`。

| 项目 | 实际结果 |
| --- | --- |
| Run | `1c6ee9bf-7ed5-490a-b2aa-d09b4de23fc3` |
| 正式 Verifier | `passed`，verdict `f6698342-9b51-4133-972a-2d6b6c4d17b2` |
| 控制 / 物理子步 | 485 / 1,940 |
| 原始事件 / observer 图像 | 1,015 / 97 |
| 模型等待结果 | 十条，窗口为一秒与九十秒 |
| 执行中的查询 sequence | 6、120、280、441 |
| 确认边界 sequence | 75、80、85、435、485、485 |
| 原始模型 RGBD | 三次实际观测，三项目标 |
| 当前目标跟踪查询 | sequence 80、85 |
| 表面点 standoff / 最大误差 | 0.4 / 0.0053368507 米，容许误差 0.05 米 |
| 最终零 twist 保持 | 50 个控制步 |
| 正式目标 | 距 `ball_0` 中心约 0.42718 米，要求 0.5 米；保持 97 步，要求五步 |
| 姿态 | 高度约 0.11665 米，tilt 约 0.00826 rad，满足原生要求 |
| 终止与资源 | `ended`、`device_confirmed`、`policy_stop`；session 关闭，CPU worker 数量归零 |

公开安装版 `omd validate replay` 同时要求停止进度、目标跟踪、模型来源与运动等待。
等待记录检查原始模型参数、工具身份、当前进度、原生边界身份及 `boundary_at`；
九项保存资料的无效修改全部拒绝。原始 RGBD 测量独立复算通过，
54 项无效测量修改和七项来源修改全部拒绝。

远程 perception supervisor 退出状态为 0，端口关闭。
服务 SSH 管理连接的退出状态为 255；私有总驱动记录保留 `passed=false`。
完整任务验收依据原生 task verdict、安装版严格回放与原始资料检查。
长时间 SSH 管理连接的生命周期具有独立验收要求。

## MP4 与安装检查

视频包含实际 observer 场景、head RGB、公开 agentic trace、工具参数、测量和正式结果。

| 项目 | 结果 |
| --- | --- |
| 编码 | H.264，1920 × 1080，10 FPS |
| 时长 / 完整解码帧数 | 157.5 秒 / 1,575 |
| 视频 SHA256 | `b44f1391c302c17cdcbe50ab69a65e3451619ad5b768f8c5f9f1b8b584cbafc0` |
| Renderer SHA256 | `103426b21794847614d861eec7ccbe2fce388bc2e1323a4c2eed57276e8d58f6` |
| 图像与文字 | 五张关键 PNG 数值解码、原始事件与 manifest 摘要、全部编码帧文字范围检查通过 |
| 安装 | Mac 与 Linux 各 456 个 source/resource 文件、三份许可证、34 项 checkout 之外的实际 CLI 调用 |
| 参数、场景与导入检查 | 固定安装检查源码的 101 项检查通过 |

## 资料与公开复核

资料位于 `outputs/acceptance/`，保留全部原始输出：

| 内容 | 路径 |
| --- | --- |
| 独立完整检查 | `model-motion-wait-handoff-20261013-01.json` |
| 实际 CPU 动作、原始记录与独立检查 | `model-motion-wait-native-linux-20261013-02/` |
| 完整模型任务、事件、模型传输与相机 | `model-motion-wait-mac-ssh-20261013-02/` |
| MP4 | `model-motion-wait-mac-ssh-20261013-02/agentic-motion-wait-01.mp4` |
| 原始模型观测与服务退出 | `model-motion-wait-perception-linux-20261013-02/` |
| 原始 RGBD 独立复核 | `model-motion-wait-rgbd-audit-mac-20261013-02/result.json` |
| 安装版严格回放 | `model-motion-wait-installed-replay-mac-20261013-02.json` |
| 等待记录与九项拒绝 | `model-motion-wait-records-audit-mac-20261013-01.json` |
| Mac 安装 | `model-motion-wait-audit-package-mac-20261013-02/result.json` |
| Linux 安装 | `model-motion-wait-audit-package-linux-20261013-02.json` |
| 101 项检查 | `model-motion-wait-imports-20261013-05.xml` |

公开命令与实现入口见[运动等待](../native-motion-wait.md)。
GPU、Newton 当前源码执行、真机、自主移动目标跟随和独立语义识别准确率继续分别验收。
GPU 验收与 RL 保持停止。用户原有 `environments/isaac-assets/uv.lock` 未修改。
