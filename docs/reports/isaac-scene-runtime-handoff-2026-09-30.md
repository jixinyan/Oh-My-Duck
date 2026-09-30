# Isaac 场景与 policy 工具进度

日期：2026-09-30 UTC。

## 停止条件

本次最小目标已完成：在 NVIDIA Office 中执行官方 ONNX policy，通过实际 Isaac Lab/Newton/BAM 读取相机与 ToF，并测量停止结果。本次验证与资产转换进程均已关闭，GPU0 返回 24 MiB、0% 使用率。RL 训练保持停止；多场景、多 policy 完整 Demo 留待后续继续。

本次功能源码提交为 `07ece085a540cb535d8b43bd8141cec29ce705b7`。后续继续工作时读取本记录中的实际证据和待验收项目。

## Office 实际运行

验收主机为 `jd_B300` 的 NVIDIA H20G，设备 `cuda:0`，实际 solver 为 Newton `SolverMuJoCo`。Office 的 3,645 个 authored collision Mesh 和一个 ground Plane 均进入 solver，机器人包含 70 个 collision geometry。23 个镜像 mesh 保留原 geometry；逐顶点变换位置一致，转换前后世界坐标 bounding box 的最大绝对误差为零。

官方 `alpha_walking` 的命令为 `[0.2, 0.25, 0]`。初始 `velstand` 执行 12 个控制步，行走执行 100 个控制步，随后零命令执行 25 个控制步。共 137 个控制步、548 次真实 BAM 物理调用；每个控制步包含四次 0.005 秒物理步。相对于初始稳定位置，平面位移为 `0.096044898 m`。整个验证中的非地面外部接触累计为零。

停止确认来自五个连续测量样本。最终 height 为 `0.116817668 m`，tilt 为 `0.003275492 rad`，`fallen=false`；body twist 为 `[-0.000995894, -0.000168273, 0.026379455]`，均满足现有停止阈值。相机产生真实 `320×240`、`uint8` RGB，全部数值有限，像素 variance 为 `109.5537753`。observer 视频为真实 H264、`1280×720`、25 fps、2 秒。

ToF 使用实际 Newton geometry 与当前 GPU pose 的独立 MuJoCo snapshot；射线查询环境 group 0，机器人 group 3 保留在实际 physics 中。64 个 zone 中，56 条射线命中真实环境，原始距离为 `6.744284416–390.770763568 m`，其余八条没有命中。全部超过四米量程，输出 `status=255`、`distance_mm=0`。每条射线的 origin、direction、命中 geometry、原始距离与状态均已保存。此位置只覆盖环境查询和超量程语义；量程内近障碍检测仍需独立验证。真实 CPU 编译审计另外确认复制模型的 group 数组没有共享内存，修改 snapshot group 后原模型数组保持相等。

导航 point goal 的实际判定为 `goal_reached=false`。本次覆盖环境、policy 动作、传感器读取与测量停止；Isaac 原生 Harness 的真实 VLM 会话、正式任务成功和长距离导航均待验收。

远程产物目录为：

`/home/jixin/workspace/code/Oh-My-Duck/.job-sources/isaac-scene-runtime-20260929/outputs/isaac-runtime/office-alpha-11/`

其中保存 `result.json`、`trajectory.json`、`stop-trajectory.json`、head RGB 和 `observer.mp4`。本地结果、停止轨迹和视频另保存于 `.cache/isaac-scene-compatibility/office-alpha-11/`。真实编译审计保存于远程同级的 `allcollisions-compiler-audit-02.json`，本地副本在 `.cache/isaac-scene-compatibility/`。

Isaac Lab 固定为 `v3.0.0-beta2.patch1@ffff603eafc6b74264a5261cc0183d6a65390d78`；运行环境使用 Torch `2.10+cu130`、IsaacLab `6.1.14`、Newton `1.2.1`、Warp `1.13.0`、mujoco-warp `3.8.0.3` 与 ONNXRuntime `1.29`。完整机器人转换 fingerprint 为 `c4fda72f7116724cdfb02e6b928de59bf4e625162f4cdc07ebb5b081b301a787`；Office 来源 manifest SHA256 为 `1093dcbb0a25799fef75bbdce34fd2f82ed7a70cd51cc5c339993fa0d484c681`。转换 fingerprint 包含当前 `environments/isaac-assets/uv.lock`；该文件的现有用户修改保留，不纳入本次源码提交。

实际执行的 `isaac_official.py` SHA256 为 `02cd813c7bab1eb596b972bef4c4f98e3015b9f128fb9ef219eccdad79588991`，验收脚本 SHA256 为 `2b2137de186fdf19804c9b44ed38e743d83b3ad40415a3e6b155c9e6885d5fc2`，lock SHA256 为 `ea626b92d8a428575142e877e94a1412b1e3666874fc45a65cbac3791289ef65`，最终 `result.json` SHA256 为 `37511205f4899941b4f575c37593c19b85a74fe6aa23a9bdd14b43b8c8a6d6c6`。本地源码与结果副本的 SHA256 已分别复核相等。

重复运行需要完整场景资源、对应转换 fingerprint 和新的输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../src ../.envs/isaac-newton/bin/python accept_isaac_official_backend.py --scene-config ../configs/simulation-demo/office.json --catalog /home/jixin/workspace/code/Oh-My-Duck/.cache/official-policies-runtime/1b56c396825c052a4e26e95cf2b8d8298af9e9b4 --output ../outputs/isaac-runtime/office-alpha-next --steps 100
```

工作目录为远程源码副本的 `scripts/`。本次运行使用远程 `.cache/office-scene.json` 中相同的 Office 来源、出生位置与目标配置。

独立 Isaac 机器人配置使用原始官方 `robot_allcollisions.xml` 的 collision 属性。真实编译审计确认共享 robot factory 的 contact、material、质量、惯性与 armature 完全保持原定义；独立场景配置的质量、惯性与 armature 同样完全一致。场景配置保留原始 servo `condim=3`，参数差异逐项保存在审计中。Office mesh 接触采用当前 SolverMuJoCo 的 convex geometry；门口和凹形区域的实际通行性仍需验证。

## 已验证的原生 policy 工具

官方策略固定为 `pollen-robotics/microduck-policies@1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。2026-09-30 查询 Hub API 时，其 main 仍指向该提交，共十个 ONNX。原生 EDH 固定为 `8a5e685b22d032207f53db20454f0992a4ad60fd`。

原生工具支持 `perpetual`、`scripted` 与 `episodic` manifest 类型。`select_policy` 要求 Gate 确认动作边界和至少五个测得的停止样本。`transition_policy` 在完整 episodic 执行产生 `episode_terminated` 后明确接续下一项 policy；它保留实际物理状态和上一动作，停止结果仍由后续测量判定。

真实 CPU MuJoCo/BAM 验证通过以下流程：初始 `velstand` 执行 75 个控制步，`kick_left` 执行全部 25 个控制步，原生 Gate 产生终止边界；随后经工具接续 `alpha_stand`，启动新的原生 execution 并执行 100 个控制步。转换前后的 episode、sequence、仿真时间、body position 和 body twist 完全一致。最后测得 90 个连续停止样本，`finish_policy` 接受停止结果。总计 200 个控制步、800 个物理子步。

最终共享 robot factory 下的重复验证结果为 `.cache/isaac-scene-compatibility/cpu-kick-left-transition-03.json`，同样获得 90 个连续停止样本；此前结果保存在 `cpu-kick-left-transition-02.json`。这项检查覆盖原生工具、真实 ONNX、动作接纳、完整时长、状态保留和测量停止。它没有调用 VLM，也没有验证球体位移。

原有 CPU 命令与停止路径也完成实际回归：75 个初始步、100 个行走步和 100 个零命令步；测得 32 个连续停止样本，外部障碍接触累计为零。结果为 `.cache/isaac-scene-compatibility/cpu-native-command-stopped.json`。CPU server 的 `/api/config` 已检查原有公寓 profile；该检查只覆盖启动和公开配置。

## 场景导入接口

`omd harness --scene-config <json> --simulation-python <python>` 读取外部场景配置，并使用原生 EDH 会话、ActionGate、execution tools 和独立 Verifier。配置中的资源路径相对于项目根目录解析。完整 server 与 GPU worker 在远程服务器运行时，可以通过 SSH 转发 server 端口访问。

配置包括 `backend=isaac-newton`、`scene_id`、`usd_path`、`provenance_path`、`public_map_path`、`device`、`spawn_pose`、`goal`、`task_instruction` 和 `budget`。当前导航目标使用 `kind=point`、`target_xy_m`、`distance_m` 和 `hold_ticks`。场景、机器人、policy 命令与正式目标判定分别保留在相应模块。

`configs/simulation-scenes.json` 提供 Office、Hospital 与 Warehouse 的来源目录。资源校验、原始 USD、许可和几何资料见 [场景资源报告](isaac-scene-assets.md)。下载文件使用 `data/scenes/`，该目录由 Git 忽略。公开代码包含获取工具与配置。

## 后续验收

- Isaac 原生 Harness 的真实模型会话与正式 Verifier 结果需要单独验收。
- ToF 量程内障碍距离、近障碍暂停与凹形 geometry 通行性需要实际验证。
- Hospital 与 Warehouse 需要完成各自来源清单、出生位置、任务配置和物理检查。
- 长距离导航需要记录实际路线、障碍接触、转向、目标保持和停止结果。
- 多 policy Demo 需要分别验证坐站、head/body 控制、翻滚恢复、拾取动作及踢球结果。
- `roller` 与 `crouch` 需要对应的带轮机器人模型。

## 重复 CPU 验证

```bash
TMPDIR="$PWD/.cache/tmp" PYTHONPATH="src:$PWD/.cache/edh/8a5e685b22d032207f53db20454f0992a4ad60fd/harness/physical-runtime/src" .cache/cpu-apartment-locked-venv/bin/python scripts/accept_harness_policy_transition.py --policy kick_left --output .cache/isaac-scene-compatibility/cpu-kick-left-transition-next.json
```

每次验证需要新的输出路径。源码中的 `scripts/accept_harness_policy_transition.py` 也接受 `--scene-config`、`--policy-dir` 和 `--edh-source`，用于后续真实 Isaac 原生工具验收。
