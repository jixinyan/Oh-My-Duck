# 原生目标跟踪验收

日期：2026-10-12，America/Chicago。`microduck.read_track` 已接入原生 Harness。
实际 CPU 模型任务完成观察、查询同一目标、接近、重新观察、停止和正式 Verifier 检查。
独立 CPU 感知测试完成原生移动物体与 YOLO26 连续图像关联。

## 来源与运行环境

| 项目 | 来源 |
| --- | --- |
| 实际模型任务与 SSH worker | `c7792da3a8e45054d58a7603cb687aef1b67cc63` |
| 原生移动物体与实际 YOLO26 测试 | `c1359985af4bab9dd0c3f6bd06bfa27b65cf9957` |
| 独立复核与两台主机的安装包 | `8f9d0f9f4ba54341c9e789be56f756536ea90028` |
| 视频 renderer | `e4df276`，SHA256 `af429c58133ecfe25073382b2f8634be6fa99eede36302b6da6dd8112eaf5d05` |
| EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| YOLO26 权重 | `yolo26s.pt`，SHA256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| 实际模型 | OpenAI `gpt-6-luna`，Responses，high，8192 tokens 输出预算 |
| 物理与渲染 | Linux CPU MuJoCo/BAM，OSMesa / llvmpipe，`CUDA_VISIBLE_DEVICES=''` |

以上版本的生产 `tracking.py`、原生 `session.py`、感知 `service.py` 与 Node 工具
实现内容相同。凭证由私有配置读取到进程内存；源码、安装包和公开报告保留模型设置。
标准 61 个 actor 观测、14 个动作、13 个 command、50 Hz 和四个 0.005 秒物理子步保持原样。

## 模型任务与视频

公共场景为 `configs/simulation-demo/apartment-target-tracking.json`：官方 8 × 6 米
公寓，起始 `(-3.65, 0, 0)`；目标中心半径 0.5 米、upright 与五个保持控制步，
预算 1000 个控制步 / 600 秒。目标接近距离为观察表面点的 0.4 米。

| 测量 | 结果 |
| --- | --- |
| Run | `5cd97d2f-498f-40be-a06e-2f1749745162` |
| 独立 verdict | `3f5391df-4392-4c52-a32a-b7281d1139ec`，`passed` |
| 任务与停止 | `succeeded`，确认的 `policy_stop` |
| 控制 / 物理子步 | 223 / 892 |
| 原始事件 / observer 图像 | 564 / 44 |
| 完成的模型 Responses | 42 |
| 实际 `read_track` 调用 | 三次，同一目标在 sequence 75 和 173 的当前测量 |
| 感知来源 | `simulator_ground_truth` |
| 目标距离误差 | 0.004715480830564556 米 |
| 独立停止进度检查 | 九项，原有五个实际停止样本要求 |
| 资源释放 | session 释放，所属 worker 数量为零，原生服务正常退出 |

MP4 显示实际 observer 场景、head RGB、公开 Planner 文字、工具查询、目标测量与
正式 verdict。1920 × 1080，H.264，10 fps，97.5 秒，975 帧全部完整解码通过；
每帧文字边界、原始事件和图片来源摘要通过核查。

视频：`outputs/acceptance/target-tracking-model-mac-ssh-20261012-01/agentic-target-tracking-01.mp4`。
SHA256：`827cc7157e5f6c471ecbb6f9560bef29ae2d6c48712eeec9d18246db00948045`。

## 原生感知与关联

| 检查 | 结果 |
| --- | --- |
| 原生实例跟踪 | `ball_0` 保持同一 `track_id`，175 控制步 / 700 子步 |
| 实际物体运动 | 施加 0.003 N world X 力；3D 位移 0.09834984019662621 米，水平位移 0.06895400690334 米 |
| YOLO26 CPU 关联 | 250 控制步 / 1000 子步；参数设备为 CPU，CUDA 未初始化 |
| 目标离开与返回画面 | `missing` → `tentative` → 两次新测量后的 `current` |
| 原始感知记录 | 十次 capture，保留实际 RGBD、模型原始 response、实例 segmentation 和物理 sequence |
| 独立实例像素检查 | 三项确认观测；关联目标对应原生 `ball_0` 和 `liv_table` |
| 重复图像读取 | 三项记录，确认次数与原始测量时间保持不变 |
| Native query 检查 | 十次实际查询；当前与历史记录分别保留原始时间、sequence、来源与目标 |
| 原生无效查询 | 每个来源四项参数拒绝，共八项 |
| 改动后的测量数据 | 二十项 sequence / 几何改动被拒绝 |
| 官方 ONNX 独立复算 | 425 次动作、61 / 14 / 13 维数据，最大误差为零 |
| 原始 observer 图像 | 85 张，完整 PNG 解码且像素非均匀 |
| Task 生命周期 | 两个来源均完成停止、关闭、创建新任务及旧目标查询拒绝 |

YOLO26 保留原始类别，所选择目标的 `label` 为 `orange`；对应实例像素为 `ball_0`。
这里的检查覆盖测量与连续关联。识别类别准确率需要独立标注评估。
同一物理 sequence 的原生更新可以包含重新采样的 ToF 噪声；复核使用接纳动作、
原生身体状态和观测来源分别检查控制与感知。

## 安装与独立复核

Mac 与 Linux 分别构建、安装 wheel / sdist；每台主机核验 454 个源码与资源文件、
三份 license 和 33 项公开命令。Linux 安装包使用声明的 `validation` 依赖，
在项目目录之外完成原始跟踪记录、RGBD、实例像素与 ONNX 动作复核。
Mac 安装包在项目目录之外完成正式模型任务的 replay 复核。
最终源码的场景与导入检查通过 79 项，Node 工具文件通过语法检查。
原始模型查询的八项数据改动被独立检查拒绝；五张视频关键帧通过 PNG 与来源核查。

| 记录 | 路径 |
| --- | --- |
| 模型、事件、正式结果和资源释放 | `outputs/acceptance/target-tracking-model-mac-ssh-20261012-01/` |
| 原始 CPU 感知与物理记录 | 远程 `outputs/acceptance/target-tracking-native-linux-20261012-03/` |
| 原始模型服务请求 | 远程 `outputs/acceptance/target-tracking-driver-linux-20261012-03/inspections/` |
| 独立测量、关联和 policy 复核 | `outputs/acceptance/target-tracking-audit-linux-20261013-02.json` |
| Linux 安装包完整复核 | 远程 `outputs/acceptance/target-tracking-installed-audit-linux-20261013-02/` |
| 两台主机安装检查 | `outputs/acceptance/target-tracking-final-package-{mac,linux}-20261013-01/result.json` |
| 视频与查询改动检查 | `outputs/acceptance/target-tracking-video-query-audit-mac-20261013-01.json` |
| 独立汇交检查 | `outputs/acceptance/target-tracking-handoff-20261013-01.json` |

这组结果验证查询工具、动态原生实例、YOLO26 图像关联和一个正式模型接近任务。
自主跟随、模型感知来源的完整目标任务、物体交叉与长时间遮挡、标注准确率、
Newton 执行和真机任务保留各自的验收要求。GPU 验收与 RL 保持停止。
公共接口、参数与源码位置见[目标跟踪说明](../native-target-tracking.md)。
