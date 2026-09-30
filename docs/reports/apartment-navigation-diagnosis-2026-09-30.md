# Office 门口导航诊断

运行条件：官方 `scene_apartment.xml`、官方 `alpha_walking.onnx`、官方 CPU MuJoCo/BAM 控制器、固定 policy revision `1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。各诊断从原生位姿 `(0, 0, 0)` 和 seed `20260929` 开始，按照对应记录运行 75、182 或 187 个 `velstand` 控制步；控制频率为 50 Hz，每个控制步包含四个 0.005 秒物理步。office 目标使用同一原生 room GT 和五个连续控制步保持条件。诊断没有修改场景、目标、出生位姿、碰撞、模型权重或控制器。

## 原始任务证据

原生任务 `6b8b5c45-2f8a-45fa-8d28-e95a7465c668` 的官方导出位于 `.cache/demo/6b8b5c45-2f8a-45fa-8d28-e95a7465c668/`。机器人在 `twist=[0.4,0,0]` 下前进至约 `(0.60,0.06)` 米，随后长时间停留在 office 入口附近；独立 Verifier 给出 `goal_reached=false`，原生动作门因 `budget_exhausted` 终止。重放同一策略、命令和控制步数的真实物理诊断在 `sequence=592` 测到 `off_cabinet` 接触，在 `sequence=725` 测到五个该物件接触点；最终原生任务位置约 `(0.582,0.022)` 米。诊断数据保存于 `.cache/apartment-collision-diagnosis-01.json` 和 `.cache/apartment-collision-diagnosis-contact-02.json`。

模型里的 office 门位于 `x=0.5` 米，门口 `y=[-0.2,0.6]` 米。`off_cabinet` 的碰撞范围为 `x=[0.67,1.03]`、`y=[-0.18,0.18]` 米。`off_shelf` 的碰撞范围为 `x=[0.6,1.4]`、`y=[0.76,0.96]` 米。这些数值由 MuJoCo 编译模型的具名静态碰撞 geom 计算；`public_scene_info()` 同时导出六处门口和 57 个静态障碍物。

官方 CPU inference 在 `src/oh_my_duck/rl/evaluation/rehearsal/infer_policy.py` 明确以机器人局部坐标比较速度命令和实际速度。CPU backend 同时保存 `body_twist` 与 `body_twist_world`，诊断轨迹记录了两个坐标系和实际 yaw。直行时 `y` 很小，碰撞物件为 `off_cabinet`，因此路径主要受门后家具占据的空间约束。

## 同出生位姿的实际运动

- `twist=[0,0.25,0]` 连续 200 个行走控制步，`y` 仅增加约 `0.0044` 米。原地转向 `twist=[0,0,0.7]` 连续 100 步，yaw 仅增加约 `0.1025` rad。诊断保存于 `.cache/apartment-route-diagnosis-01.json`。
- `twist=[0.2,0.25,0]` 第 498 个行走控制步达到原生 office GT，总 `sequence=685`，位置 `(0.70674,0.29545,0.12026)` 米。继续行走至第 600 步，总 `sequence=787`，位置 `(0.87003,0.29968,0.12051)` 米。随后零命令 50 步，总 `sequence=837`，位置 `(0.86105,0.29821,0.11688)` 米，倾角 `0.00578` rad，`goal_reached=true`，保持计数 `157/5`。整个 187+600+50 控制步内，非地面外部接触累计为零。完整延长诊断保存于 `.cache/apartment-after-goal-diagnosis-01.json` 与 `.cache/apartment-after-goal-contact-02.json`。
- `twist=[0.2,0.3,0]` 第 468 个行走控制步达到原生 office GT，总 `sequence=655`，位置 `(0.71685,0.28728,0.11679)` 米；到达前没有非地面接触。
- `twist=[0.3,0.3,0]` 在第 259 个行走控制步首次接触 `off_cabinet`，总 `sequence=446`，位置 `(0.59398,0.22617,0.11794)` 米，接触法向力约 `2.208 N`。三组对角命令及每个控制步的测量保存在 `.cache/apartment-diagonal-diagnosis-01.json`。

以上结果属于相同确定性条件下的物理诊断。正式 agent run 需要以原生动作门、传感器事件、停止边界和独立 Verifier 再次验收。

## 75 步站立准备后的恢复能力

原生动作门使用 75 个 `velstand` 控制步后切换 `alpha_walking`。同条件 CPU 诊断复现 `twist=[0.2,0.25,0]` 在总 `sequence=423`、位置 `(0.49826,0.18689)` 米时触发中央 4×4 ToF 的 90 mm 停止条件；当时最近有效距离为 `89 mm`，非地面外部接触累计为零。诊断保存于 `.cache/apartment-warmup-75-diagnosis-01.json`。

从该物理状态直接执行 `twist=[-0.2,0.3,0]`，150 个控制步后到达 `sequence=573`、位置 `(0.42638,0.38503)` 米；中央最近有效距离增加到 `269 mm`，非地面接触仍为零。接着执行 `twist=[0.2,0.25,0.4]`，193 个控制步后到达 `sequence=766`、位置 `(0.71119,0.43136)` 米，原生 office GT 满足，中央最近有效距离 `881 mm`。零命令再运行 50 个控制步，最终 `sequence=816`、位置 `(0.70486,0.42249,0.11697)` 米，机器人局部线速度各分量绝对值均小于 `0.0004 m/s`，yaw 速率绝对值约 `0.01081 rad/s`，倾角约 `0.00488 rad`；原生 GT 保持 `55/5` 个控制步，非地面接触累计始终为零。完整数据保存于 `.cache/apartment-recovery-commands-stop-02.json`。

同一恢复状态使用 `twist=[0.15,0.3,0.4]` 也达到 office GT，零命令 50 步后位置 `(0.71462,0.43187)` 米、GT 保持 `55/5`、非地面接触累计零。`twist=[0.2,0.3,0.7]` 到达 GT 后，零命令阶段最终位置 `x=0.69406` 米，GT 未保持。恢复后继续使用 `twist=[0.2,0.25,0]` 在 `sequence=705`、位置 `(0.61149,0.32321)` 米再次触发 `87 mm` 距离停止，累计非地面接触为零。带正 yaw 的运动命令在这些实际轨迹中产生了更合适的过门路径。

75 步与 187 步的站立状态位置、yaw 和倾角接近，关节角最大差约 `0.00224 rad`；最大关节速度分别为 `0.01194 rad/s` 与 `0.00039 rad/s`。一个连续 25 步的站立测量窗口在 `sequence=182` 满足局部线速度、yaw 速率、倾角、关节速度与关节角跨度限制，随后相同 `[0.2,0.25,0]` 仍在门口触发 `87 mm` 距离停止。该测量条件能够描述站立收敛，单独使用它无法保证穿门轨迹。数据保存于 `.cache/apartment-standing-75-187-01.json` 和 `.cache/apartment-transfer-diagnosis-01.json`。原生运行需要持续读取 ToF、位姿与接触证据，并在停止边界选择新的有限动作段。

## ToF 与接触测量

原始 8×8 ToF 状态和距离继续直接来自 MuJoCo ray，诊断只额外记录首个命中的 geom。`status=5` 代表有效命中；没有四米内目标的区域保留原始无目标状态。采样位置里没有自身 geom 首次命中；接近文件柜时，前向区域首先命中 `off_cabinet`。运行时没有过滤自身命中，诊断将 `self_occluded`、`external_first_hit`、`no_target_within_4m` 分开记录。

中央 4×4 有效区域的逐控制步距离显示：`[0.2,0.25,0]` 到达前最小值始终至少 `101 mm`，10% 分位数至少 `106.5 mm`，中位数至少 `144 mm`；`[0.2,0.3,0]` 的对应最小值为 `73/82/97 mm`。发生碰撞的 `[0.3,0.3,0]` 在接触时分别为 `88/94.5/140 mm`，并在接触前 22 个控制步记录过 `66 mm` 的中央最小距离。这些轨迹说明统一距离阈值无法同时容许全部已测成功命令并阻止已测碰撞命令。中央最小距离 `90 mm` 用于已测 `[0.2,0.25,0]` 运动时，测得最小余量为 `11 mm`；正式运行仍需依据原生接触和实际速度监控。

`CpuMujocoBamBackend.observe_control().measurements.contact_evidence` 在每个实际物理子步读取 `mj_contactForce`。`non_ground_external_contact_samples_total` 统计正法向力的非地面外部接触样本，`non_ground_external_contact_control_steps` 统计出现这些接触的控制步，`first_non_ground_external_contact_sequence` 保存首次控制序号；`current_control_non_ground_external` 包含每个 geom 的样本数与最大法向力。地面和自身接触分别使用 `current_control_ground_contact_samples` 与 `current_control_self_contact_samples`。重复读取观测不会增加计数。直行撞柜诊断到 `sequence=1205` 时，累计 442 个非地面接触样本、94 个接触控制步，首次接触为 `sequence=356`；安全对角路线到 `sequence=837` 的累计数为零。

## 复现命令

在项目根目录使用锁定的 CPU 环境，设置 `TMPDIR="$PWD/.cache/tmp"` 与 `PYTHONPATH=src`，运行：

```sh
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_collision.py --output .cache/<new-collision-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_routes.py --output .cache/<new-routes-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_diagonals.py --output .cache/<new-diagonals-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_after_goal.py --output .cache/<new-after-goal-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_warmup.py --output .cache/<new-warmup-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_transfer.py --output .cache/<new-transfer-output>.json
.cache/cpu-apartment-locked-venv/bin/python scripts/diagnose_apartment_recovery_commands.py --output .cache/<new-recovery-output>.json
```

输出目录必须使用新文件名，诊断脚本拒绝覆盖既有证据。
