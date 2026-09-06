# Headless 仿真：首轮执行计划

日期：2026-09-06。状态：用户已批准执行，环境安装与验证进行中；实际结果见 [implementation-status.md](implementation-status.md)。本文件细化原执行计划第 01–02 步，不代表完整源码审计已经通过。

## 最新执行优先级

用户已补充：先建立服务完整 Agentic Microduck scope 的整体框架、模块边界与入口，再逐步实现功能。本 headless 计划保留为后续训练后端工作单；当前先验收框架。首轮源码、模型获取和环境下载已启动，尚未提交 GPU 任务。

## 用户已确认的条件

- 当前服务器是开发入口，训练通过 job 提交，使用 headless 路径。
- 现阶段没有真机，聚焦仿真；最高验收标记为 `sim_validated`。
- 外部 Harness 尚未就绪，后续接入使用确定性 mock。mock 只替代上层协议对端，运动仍由真实仿真和策略执行。
- 两个官方仓库参考默认分支。每次实验记录具体 SHA，避免默认分支后续更新改变同一次对照。

## 已查明的环境与版本

| 项目 | 只读检查结果 |
|---|---|
| 调度入口 | `/usr/local/bin/submit`，Alaya HTrain；`squeue` 是其包装器，不是标准 Slurm 安装 |
| 可用项目 | `agent`，也是当前默认项目 |
| 提交参数 | `--name`、`--nodes`、`--gpus-per-node`、`--cmd`、`--log-path`、`--project` |
| GPU 请求 | 支持每节点 1、2、4、8 GPU；submit 默认是 8，必须显式填写 |
| 当前会话可见硬件 | 8 × NVIDIA H200，每张报告 143771 MiB，驱动 580.126.20；不能据此认定 job 已获分配 |
| 当前解释器 | Python 3.13.13；uv 0.11.25 |
| 参考 RL 解释器要求 | Python `>=3.12,<3.13`，需独立环境 |
| 持久路径 | `/mnt/data/users/heyang/workspace/code/oh-my-duck`；当前会话下与 `/home/heyang/workspace/code/oh-my-duck` 的设备号、inode 一致；worker 可见性尚待 job 验证 |
| 当前工程 | 只有设计文档，尚未初始化 Git |

上游参考快照已下载至 `/tmp/oh-my-duck-audit-20260906/`，只用于调研，不作为 worker 的持久依赖。

| 来源 | 默认分支 / 修订 | 文件清单规模 |
|---|---|---|
| `pollen-robotics/microduck_rl` | `develop@29e887ecfbf5d37144759e5a9f8a176dfb83d547` | 231 个 Git 跟踪文件 |
| `pollen-robotics/microduck` | `main@bc41fb5c9a9b39894669c1e022e375cf83800382` | 283 个 Git 跟踪文件 |
| `pollen-robotics/microduck-policies` | HF `088524a64e2557dc453256b6071dbb9d23888802` | 公开、无需申请访问；存在 `alpha_walking.onnx` 和 schema 2 manifest |

文件数量来自完整 `git ls-files` 清单，不代表逐文件内容均已审读。当前已追踪运动关键链路，并定点检查传感器、媒体与仿真服务；完整阅读覆盖表及非关键模块审读仍属于第 01 步待完成工作。

RL 的 `uv.lock` 固定：mjlab 1.3.0、MuJoCo 3.10.0、MuJoCo-Warp 3.8.1、Warp 1.12.0、RSL-RL 5.0.1、x86_64 PyTorch 2.9.1；BAM 为 `62bd8ce12154340be97e06f7f41a0ca8f116d967`。这些是参考环境版本，不直接用作 Isaac 环境版本。

Isaac 候选为 `v3.0.0-beta2.patch1`，优先 Newton / MJWarp 的 kit-less 路径。官方仍将 Newton 支持标为 Beta；精确依赖组合及 BAM 所需 solver 字段尚未完成验证。先核查固定 tag 的依赖与示例，再提交独立兼容性任务，不混用在线滚动文档和不同版本的依赖。

## 会影响实现的源码发现

1. **CPU 回放有 BAM，但入口尚非 headless。** 当前 RL 默认分支 `scripts/infer_policy.py` 提供 `load_bam_model`、`load_mujoco_with_bam`、`PolicyInference`。其主循环进入终端输入和 `mujoco.viewer.launch_passive`，没有 headless 参数。应复用这些物理与推理实现，补有限步数、固定命令输入和输出记录的入口。仅设置 EGL 不会把窗口循环变成离屏回放。
2. **观测格式必须显式对齐。** 当前回放仍保留旧 51 维模式；61 维路径需 `new_cmd_obs=True`。迁移基线使用身体角速度、投影重力、14 维关节相对位置、速度、原始上一帧动作及 13 维命令；嘴部独立处理。具体实现见 runtime `duck-control/src/obs.rs`、`model.rs` 和 RL 回放脚本。
3. **执行器不是通用 PD。** `friction_dr_bam.py` 与 `tasks/mdp.py::expand_bam_friction_fields` 涉及每环境的 `dof_frictionloss` / `dof_damping`。还需读取锁定 BAM 依赖，核对延迟计数所处时钟、外部负载、电压降、缓存更新和 reset；不能把 lag 数字直接当作 50 Hz 控制步。
4. **训练、回放和 runtime 调参分别记录。** 训练 action scale 为 1.0；runtime `Tuning::default()` 的行走 scale 为 0.9，并有滤波配置。后续以实际生效配置对照，不能把默认常量当作所有模式的运行值。第一份仿真报告分别记录训练条件和部署回放条件。
5. **现成 body server 不能作为 BAM 基线。** `sim/body_server.py` 支持 `--headless`，但直接加载 XML 位置执行器并修改 PD 增益，未走上述 CPU BAM 控制器。其文档所述 `duck_control::sim` / `robotd --sim` 在锁定 runtime 默认分支中未找到。该路径暂不作为本轮复现依赖，也不切换到未获选择的其他 runtime 分支。
6. **导出会受项目补丁影响。** 当前 `scripts/export.py` 包装 `src/mjlab_microduck/export.py`；后者调用 runner 的 ONNX exporter 并附加元数据。`tasks/mdp.py` 在导入时还修改奖励、PPO returns 和排除被动关节的元数据生成。迁移时需要逐项审查，不能仅复制 exporter 或通过 `nan_to_num` 后的值推断物理过程没有 NaN。
7. **公开 ONNX 可用于先行基线，不等于拥有训练 checkpoint。** 先检查官方权重形状、元数据和数值，再回放；短训练另行产生 checkpoint，用于验证保存、恢复和导出。公开 manifest 未证明权重与当前训练配方来自同一提交，需保留这一证据边界。

## 拟执行顺序与交付

### A. 完成第 01 步审读

- 生成 514 个文件的清单，区分完整阅读、定点阅读、仅列入清单、资产、生成数据及外部依赖；审读关键依赖和剩余模块。
- 形成训练 → 资产 / BAM → 观测 / 动作 → reward / reset → PPO / checkpoint → ONNX 的源码定位表。
- 形成 runtime 策略加载 → 观测构建 → 动作映射 / 滤波 → 总线写入的定位表；补传感器与媒体接口覆盖。
- 输出精确关节表、HOME、物理与控制时序、随机化、reset、导出元数据和模型来源记录。
- 检查资产导入后的碰撞与闭环约束；区分 `robot_walk.xml` 的训练编译配置与 CPU `scene.xml` 的碰撞模型。
- 保存代码、3D 资产、模型和第三方驱动的来源及许可声明；这里只记录上游声明，不将代码许可证推及全部资产。

通过条件：迁移关键项能定位到实现；未验证项有具体检查方法。不能把文件清单生成成功算作完整审读成功。

### B. 建立独立官方参考环境

拟使用工程内 `reference/` 保存固定快照或其获取记录；参考 Python 3.12 环境、缓存与后续 Isaac 环境分开。源码变更在本项目的适配层或可记录补丁中管理。

- 按锁文件安装，记录实际包版本和环境清单。
- 下载固定 HF 修订的行走 ONNX、manifest 和来源说明，计算 SHA-256。
- 日志首先写本地文件 / TensorBoard；不依赖交互式 W&B 登录或云端日志上传。
- 运行必要的配置、关节映射与 CPU BAM 检查；GPU 初始化与训练进入任务系统。

通过条件：新进程可导入全部需要的模块，版本符合锁定记录，权重契约可检查。

### C. 第一批 job：先验证环境，再短训练

已批准：项目 `agent`，先以单节点单 GPU 验证；必要时使用单节点多 GPU。用户明确取消此前提议的 3 个任务 / 每个 30 分钟限制，不添加人为时长或数量上限。任务仍需有明确实验目标、退出条件、独立输出和资源记录。

1. **Worker 环境探测**：检查持久路径、解释器、已分配 GPU、CUDA / Warp、无显示器仿真步进，以及 EGL 离屏渲染一帧。当前入口机 GPU 信息不能替代本项。
2. **训练 smoke**：`Mjlab-Velocity-Flat-MicroDuck`，64 个环境、5 次迭代、固定种子；关闭训练渲染，保存 checkpoint，再恢复少量迭代。核对 reset、奖励、非有限状态诊断、日志和退出码。5 次迭代只证明管线可用。
3. **导出与回放验证**：短训 checkpoint 使用官方 exporter 导出；同一批观测比较训练推理与 ONNX 输出。官方预训练策略执行无窗口 BAM 回放，保存状态、动作、指标与短视频；未学会行走的 smoke 策略不承担运动性能验收。

提交外壳拟为：

```text
submit --project agent --nodes 1 --gpus-per-node 1 \
  --name <本轮唯一任务名> \
  --log-path <工程共享路径下的任务日志> \
  --cmd <含显式工作目录与环境路径的非交互命令>
```

实际提交入口已实现为 `python omd.py submit`；具体执行记录写入 `outputs/jobs/`。job 内使用普通本地训练入口；上游 `--hf-jobs` 会提交到另一平台，不用于本服务器。

### D. 固定 headless 回放协议

- 无窗口、无终端输入、有限运行步数；命令来自配置文件和仿真时钟，不依赖墙钟播放速度。
- 数值回放独立于渲染；短视频使用离屏 renderer，记录渲染后端。视频用于检查 gait，不能替代数值指标。
- 首份候选序列：静止保持 → 低速前进 → 静止 → 原地左转 → 静止。具体速度、持续时间、初始状态、种子、跌倒定义及样本数在运行前写入配置。
- 固定行走策略、初始姿态、BAM 参数、接触模型、action scale、控制频率、观测与动作历史处理。单策略与 walk/stand 切换分别评测。
- 保存输入命令、机体速度、关节状态、实际推理输入、原始动作、目标关节值、跌倒事件和脚部接触；记录首次 JIT 编译时间与稳态吞吐。
- 第一轮测量官方基线，再决定 Isaac 对照的可接受退化；不事先编造运动性能门槛。

通过条件：使用同一配置可重新运行并得到完整记录；导出链数值检查通过；官方策略表现有明确证据和失败样本。

### E. 基线通过后进入 Isaac 迁移

按原计划顺序推进：固定 Isaac / Newton 组合及最小 solver 验证 → 资产导入 → BAM 单关节 / 接触对照 → 61/14 契约对照 → 官方策略跨仿真评测 → Isaac 短训及正式训练 → 官方兼容导出 / CPU BAM 回放。

首任务保持普通平地速度跟踪。齿隙、roller、语音、复杂视觉、真实硬件与 Harness 功能不在首批 job 中。后续 mock Harness 只验证工具请求、状态与取消协议。

## 尚待实测，而非待猜测的事项

| 不确定项 | 消除方法 |
|---|---|
| worker 挂载、驱动与分配 GPU 是否匹配入口机 | 第 C1 项探测 job |
| EGL / 离屏渲染能否在 worker 使用 | 第 C1 项保存实际渲染帧及运行日志 |
| 默认分支的完整依赖能否全新安装 | 第 B 项按锁文件创建独立环境 |
| 官方权重与当前资产 / 配方的表现是否匹配 | 第 C3 / D 项固定版本回放；失败先定位 |
| BAM 延迟和 solver 原生字段的精确迁移方式 | 第 A 项依赖审读及第 E 项单关节验证 |
| Isaac 候选版本能否支持所需参数和执行路径 | 第 E 项兼容性任务；失败再给出具体替代组合供审阅 |

用户已批准实施、按需多 GPU 及资源策略，并将项目扩展为两个正式训练后端。暂不以静态调研结论宣称训练、headless 回放或 Isaac 迁移已验证。

## 来源

- [RL 默认分支快照](https://github.com/pollen-robotics/microduck_rl/tree/29e887ecfbf5d37144759e5a9f8a176dfb83d547)
- [runtime 默认分支快照](https://github.com/pollen-robotics/microduck/tree/bc41fb5c9a9b39894669c1e022e375cf83800382)
- [官方策略修订](https://huggingface.co/pollen-robotics/microduck-policies/tree/088524a64e2557dc453256b6071dbb9d23888802)
- [Isaac Lab 候选发布](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1)
- [Newton 后端说明](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/overview/core-concepts/physical-backends/newton/index.html)
- 服务器信息来自本会话的 `submit --help`、硬件 / 解释器版本查询及路径只读检查，尚无 worker 运行记录。
