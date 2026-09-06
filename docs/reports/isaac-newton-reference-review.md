# Isaac/Newton Microduck 参考仓库审读

日期：2026-09-06。范围：静态审读和本地契约比较；未安装或运行该仓库，未提交 GPU job。以下“源码确认”“作者报告”“待验证推断”分别表示不同证据强度。

参考仓库：[kabilankb/isaaclab-microduck](https://github.com/kabilankb/isaaclab-microduck)，默认分支 `main`，固定 commit `4310fe050b7a1b01e7b2f4bada103dea81d71fb2`。本地位于被忽略的 `.cache/upstream/isaaclab-microduck/`。这是第三方迁移参考；官方 MuJoCo 行为基线仍为 `configs/upstream.json` 中的 Pollen Robotics 版本。本次未复制第三方实现或资产进入本项目源码。

## 结论与可借鉴内容

该仓库提供了与本项目方向一致的 Isaac Lab / Newton MJWarp 接入样例。可以借鉴其模块组织、转换和诊断方法，但不能据此宣布我们的 Isaac 后端或 sim2sim 已验证。

| 主题 | 源码确认的做法 | 本项目采用方式 |
|---|---|---|
| 独立任务包 | Gym task 注册、配置和 MDP term 独立；薄 launcher 通过 `--external_callback` 委托 Isaac Lab RSL-RL 脚本 | 在 `training/isaac_newton/` 建立独立可安装任务包，通过现有 TrainingBackendRegistry 接入；不让核心应用导入 Isaac |
| Newton 显式配置 | `SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(...)))`，dt=0.005、decimation=4、num_substeps=1 | 同样显式选择 Newton/MJWarp，并在 worker 记录实际 solver 类型；禁止静默 PhysX fallback |
| 数组桥接 | `utils/arrays.py` 将带 `.torch` 的 ProxyArray 转为 Torch view | 在物理适配层集中处理类型、设备和 view 生命周期；不能假设所有状态都是普通 tensor |
| 资产构建 | MuJoCo reference dump → MJCF 转 USD → articulation 对比与站立检查；转换后校验实际输出路径 | 以固定官方资产生成缓存，保存源/产物 hash 和转换版本；先验证最小 walk 模型 |
| USD 渲染 | 转换脚本修复嵌套 rigid body 的变换继承 | 对选定版本先复现问题再引入修复；物理检查和实际离屏图像分别验收 |
| 策略结构 | 61D actor、14D action、HOME offset；critic 额外信息与 actor 分开 | 复用本项目共享契约，明确每一维和关节映射，不只检查 shape |
| 评估 | 对实际速度做 episode 累积，而不只看 reward | 统一命令序列，报告速度/转向误差、停止响应、跌倒、脚滑和视频 |
| 扩展任务 | 冻结伙伴策略的双机器人环境保持单个学习者接口 | 可作为未来任务组织参考；本阶段仍先完成单机器人行走链路 |

源码入口：[launcher](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/scripts/_launcher.py)、[任务配置](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/velocity/velocity_env_cfg.py)、[数组桥接](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/utils/arrays.py)、[转换脚本](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/scripts/convert_assets.py)、[评估指标](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/mdp/observations.py)、[冻结伙伴](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/frozen_partner_env.py)。

## 必须补齐的差异

### BAM 是首要技术缺口

源码确认：`robot/microduck_cfg.py` 使用 `IdealPDActuatorCfg`，kp=5、kd=0.1，被作者明确限定为资产转换检查用；`actuators/__init__.py` 为空。它没有实现官方 BAM m6 的负载相关摩擦、电压与延迟行为。PD 站立只能作为转换检查，不能作为两后端物理对齐的证明。

作者在 [BAM notes](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/docs/03_bam_actuator_notes.md) 中提出直接访问 Newton 底层 mjwarp model/data，读取 `qfrc_bias`、`qfrc_constraint`、上次 actuator torque 与约束行，写入每个 world 的 frictionloss/damping。这是值得验证的方案，**尚非本项目确认可用的 API**，也不是参考仓库已实现的 BAM。

下一阶段必须核查固定 Isaac/Newton 版本的实际字段路径、world/DOF 映射、字段扩展、更新时序和 reset；专门测试不同环境的摩擦写入互不串扰。底层私有访问集中在一个适配层，不扩散到任务代码。

### 相同维度不等于相同策略语义

源码确认：actor 顺序为角速度3、重力3、关节位置14、关节速度14、上次动作14、twist3、head4、body6；本地 AST 比较确认其 HOME 的14个名称、字典顺序与值均和 `src/oh_my_duck/robotics/microduck/protocol.py` 相同。

但其 action 和 joint observation 使用排除 `passive_` 的正则选择器，未显式指定官方14关节序列；其资产文档也报告 Isaac 与 MuJoCo articulation 顺序不同。因此“可直接交换 ONNX”尚缺 runtime 顺序证明。我们需要按名称建立双向 permutation，并对观测、动作、HOME、limits、导出 metadata 同时应用；用每个关节不同的哨兵值验收，不能只验61/14维度。冻结伙伴的实现直接拼接 articulation 全部 joint state，也不能原样推广到包含 passive joints 的模型。

观测延迟和 encoder bias 在参考仓库尚未迁移。官方固定版本包含 IMU 随机0–1控制 tick 延迟、joint velocity 固定1 tick延迟；BAM动作延迟需按其物理子步单独核对。不同时间基准不能混为一个 delay 参数。参考实现的两项 IMU misalignment term 各自创建随机姿态，是否应共享同一物理 IMU 误差也需与官方语义核对。

源码：[机器人配置](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/robot/microduck_cfg.py)、[观测配置](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/velocity/velocity_env_cfg.py)、[官方延迟配置](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)。

### 资产一致性检查需要比参考实现更完整

`check_asset_parity.py` 实际断言了 joint/body 集合、mass、joint limits、armature，并进行短站立检查；friction 和足部 contact 参数仅打印，未完整断言；惯量张量也没有完整比对。作者资产文档指出 XML 中的接触参数可经转换层传递，但官方训练运行时施加的 collision overrides 不会自动包含在源 XML 中。这两种来源必须区分。

我们应比对最终 solver model 的 mass、COM、惯量、joint axes/limits/armature、阻尼/摩擦、collision masks、contact参数及运行时 overrides；可借鉴 `save_to_mjcf` 诊断方法，但接口是否可用需在固定版本实测。相同 MJWarp solver 不保证相同模型、参数和调度。

作者报告 backlash 模型转换丢失全部14个额外 hinge，不能拿转换后的文件做 backlash A/B。首轮只做普通 walk，未来支持 backlash 必须单独解决拓扑并验收。

本地静态比较：参考 `joints_properties.xml`、`scene_walk.xml` 与固定官方文件字节相同，`robot_walk.xml` 不同，diff 包含材质及变换表示差异；这不直接证明物理不同，也不能代替编译后模型比较。资产继续由官方固定版本生成。

源码：[资产检查](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/scripts/check_asset_parity.py)、[作者资产诊断记录](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/docs/02_assets.md)。

### 环境、headless 与导出不能照抄命令

- 作者报告的环境为 Python3.12、Isaac Sim6.0.1.0、Isaac Lab `v3.0.0-beta2.patch1-18-g72cb3826d`、Newton1.2.1、Warp1.13.0、MuJoCo-Warp3.8.0.3、Torch2.10.0+cu128。SETUP安装命令却 checkout `v3.0.0-beta2`，不是报告的源码 revision；pyproject 是声明清单而非完整 lock。我们尚未将其软件组合认定为本机可用，也不修改现有 Isaac candidate pin。
- launcher 默认指向 `/home/chronos/IsaacLab`；Git 还跟踪一个指向作者绝对目录的 `assets/usd` symlink。必须使用项目配置和本地生成目录；不能照搬其机器路径。
- 参考 train/play 只是转发 Isaac Lab CLI。headless训练样例可借鉴，但本项目仍要通过服务器 job执行。视频必须单独验证离屏 renderer/camera、有限步退出和实际非空视频文件；本次没有证明参考仓库在本机可保存视频。
- `play.py` 依赖 Isaac Lab 的导出功能；README和注释提到的 `scripts/export.py` 在该commit不存在。归一化是否烘焙、外部 ONNX weights 是否齐全、官方 metadata 是否匹配，须以导出产物和独立推理对照验证。
- 作者记录 teardown 可能掩盖失败退出码。我们需保留 worker失败状态并检查结构化报告、checkpoint和视频产物，不能仅以 exit0判断通过；不先把某个版本的 `os._exit` workaround当作通用必需方案。

源码：[SETUP](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/SETUP.md)、[依赖声明](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/pyproject.toml)、[play入口](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/scripts/play.py)。

### 行走效果与任务范围

README称行走仍陷入站立，但最新 reward源码注释又描述了能前进却转向控制较差的策略。两处状态不一致；本次没有取得或复现其checkpoint，不能确认当前能力或失败根因。其reward已包含额外forward-speed、yaw误差项和不同权重，不能直接视为官方recipe等价迁移。先对齐执行器/观测/资产，再通过独立实验评估reward调整。

BallKick/BallRally可提供后续任务组织思路；该仓库把球状态限制在critic的任务设计，并不实现本项目“观察并找到任意位置红球”的感知闭环。当前不增加多机器人或踢球里程碑。

来源：[README状态](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/README.md)、[reward配置](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/velocity/velocity_env_cfg.py)、[任务注册](https://github.com/kabilankb/isaaclab-microduck/blob/4310fe050b7a1b01e7b2f4bada103dea81d71fb2/isaaclab_microduck/tasks/__init__.py)。

## 后续实施顺序与交付边界

1. `feat/isaac-newton-environment`：锁定并验证源码/依赖组合；独立环境、任务注册与job入口；实际Newton/MJWarp smoke和离屏图像。保留失败证据。
2. `feat/isaac-newton-assets`：固定官方walk资产转换、最终solver参数对比、关节映射；PD检查显式标注仅用于资产验证。
3. `feat/isaac-newton-bam`：底层solver访问与per-world字段隔离；官方BAM动力学、delay、reset对齐；单关节及负载测试。
4. 对齐观测与动作后回放同一官方策略，使用已有固定命令序列生成sim2sim指标和视频；之后再训练、恢复和导出，完成独立ONNX推理检查。

这些是既有执行计划第03–07步的细化，未扩大完整项目scope。相关实现仍为planned。每个可验证阶段小步提交；本次只提交调研、来源记录和文档同步，不把设计结论标成实现完成。
