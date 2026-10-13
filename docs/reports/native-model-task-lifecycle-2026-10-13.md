# 安装版模型任务、服务关闭与 agentic MP4

日期：2026-10-13。
完整任务、客户端、远程 worker 与安装包源码：
`932b75c6d5393a6b24704d3ea8f116b0d4fe9b52`。
固定 EDH：`8a5e685b22d032207f53db20454f0992a4ad60fd`。
官方 policy catalogue：`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。

## 实际任务与工具

Mac 在 checkout 之外使用安装版 `omd harness`。Linux 通过 SSH 运行 CPU MuJoCo/BAM、
OSMesa 和真实 YOLO26 服务。模型凭证由私有配置与进程环境提供。
框架使用固定原生 SessionEnvironment、EmbodiedBackend、ActionGate、正式任务和
独立 Verifier，保存真实模型、传感器、策略动作和物理状态。

场景为官方 8 × 6 米 apartment，出生位置 `(-1.6, -0.5, π)`。
任务目标为在 `ball_0` 中心 0.5 米范围内保持五个控制步。
原有预算保持 1000 个控制步和 600 秒。
Agent 从 `source=models` 检测中选择有效小型目标，保留原始 label、target 与 track 身份。
距离由仿真 RGBD 提供；接近参数为到当前观测表面点保持 0.4 米水平距离。

| 项目 | 实际结果 |
| --- | --- |
| Model / API / reasoning / output tokens | `gpt-6-luna` / Responses / high / 8192 |
| 实际模型 Responses | 55 次正常完成 |
| Run | `6c5a9fb3-74da-4445-9405-c33199ee9081` |
| 正式 Verifier | `passed`，verdict `62e03692-5a0a-4b08-be21-15d3ef98b512` |
| 原生 run 终态 | `succeeded` |
| 控制 / 物理子步 | 485 / 1,940 |
| 原始事件 / observer 图像 | 957 / 97 |
| 模型 RGBD / 目标 | 三次 / 三项，原始 label 为 `orange` |
| 同一目标当前查询 | sequence 80、85 |
| 等待窗口 / 返回结果 | 一秒、九十秒 / 十条 |
| 动作中等待 sequence | 6、121、126、270、442 |
| 确认边界 sequence | 75、80、85、435、485 |
| 接近误差 / 容许误差 | 0.0053368507 / 0.05 米 |
| 最终零 twist 保持 | 50 个控制步，真实停止样本满足要求 |
| 正式目标 | 距 `ball_0` 中心约 0.4272 米，保持 97 步，满足五步要求 |
| 姿态 | 高度约 0.11665 米，tilt 约 0.00826 rad，满足原生要求 |
| 原生停止 | `ended`、`device_confirmed`、`policy_stop` |

原始模型调用、测量、目标身份和暂停边界进入公开严格回放：

```sh
omd validate replay outputs/model-target/replay \
  --expected-verdict passed --expected-stop-reason policy_stop \
  --require-stop-progress --require-target-tracking \
  --require-model-perception --require-motion-wait
```

安装版检查全部通过。原始服务 RGBD、annotation PNG、工具 attachment 与 SHA256
逐项匹配，独立重新计算测量。54 项无效测量修改与七项来源修改全部拒绝。

## 服务与安装

Harness、服务 SSH 连接、远程 perception 服务与严格回放进程均以状态 0 退出。
原生 session 关闭、资源释放、CPU worker 数量归零，远程 perception 端口关闭。
私有总驱动和独立完整检查均为 `passed=true`。
所有来源副本没有 tracked 改动；GPU 与 RL 保持停止。

同一源码的 Mac 和 Linux 安装分别核验 457 个 source/resource 文件、
三份许可证与 34 项 checkout 之外的公开调用。121 项参数、场景、等待与导入检查通过。
共享 SSH 参数的有效配置、实际服务空闲 660 秒和两端端口关闭检查见
[SSH 生命周期](ssh-idle-lifecycle-2026-10-13.md)。

## MP4

实际 observer 场景、head RGB、公开 agentic trace、工具参数、测量与正式判定
在同一画面显示。视频保留真实物理运动的时间下限；原始文件与来源摘要独立核验。

| 项目 | 验证结果 |
| --- | --- |
| 编码 | H.264，1920 × 1080，10 FPS |
| 完整解码 | 1,496 帧，149.6 秒 |
| 视频 SHA256 | `8b92f15d4480c6cc51cfa23600627af0e0b8b0800765f7a72036e9c939b4b70a` |
| Renderer SHA256 | `103426b21794847614d861eec7ccbe2fce388bc2e1323a4c2eed57276e8d58f6` |
| 图像、事件与文字 | 原始 manifest、run、events、五张关键 PNG 数值解码与全部编码帧文字范围检查通过 |

## 资料位置

以下路径位于 `outputs/acceptance/`：

| 内容 | 路径 |
| --- | --- |
| 独立完整检查 | `stable-model-task-handoff-20261013-01.json` |
| 实际模型任务、原始事件、图像、传输与进程记录 | `stable-model-task-mac-ssh-20261013-01/` |
| MP4 | `stable-model-task-mac-ssh-20261013-01/agentic-stable-model-01.mp4` |
| 模型服务原始观测与退出 | `stable-model-task-perception-linux-20261013-01/` |
| 原始 RGBD 独立复核 | `stable-model-task-rgbd-audit-mac-20261013-01/result.json` |
| 严格回放 | `stable-model-task-mac-ssh-20261013-01/replay-audit.json` |
| Mac 安装 | `ssh-idle-package-mac-20261013-01/result.json` |
| Linux 安装 | `ssh-idle-package-linux-20261013-01.json` |
| 参数与导入检查 | `ssh-idle-imports-20261013-01.xml` |

GPU、Newton 当前源码执行、真机、自主移动目标跟随和独立语义识别准确率继续分别验收。
用户原有 `environments/isaac-assets/uv.lock` 保持原有内容。
