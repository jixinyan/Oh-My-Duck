# jd_B300 中断训练诊断（2026-09-26）

## 训练状态与产物

2026-09-24 08:55:31–08:55:45（服务器时间，UTC+08:00），三组训练的日志在正常更新输出后相继结束。`result.json` 均记录 `execution_failed`，完整训练阶段的退出代码均为 247；日志没有 Python 异常信息。终止原因尚未确认，三组均没有执行完整训练后的行为评估。当前服务器没有这些训练进程，已有 checkpoint 和日志保留原位。服务器报告八张 NVIDIA H20G，每张显存 275040 MiB。

| 训练 | 原始源码提交 | 已保存的原生 checkpoint | 原定更新预算 | 训练日志最后更新 |
| --- | --- | --- | ---: | ---: |
| MuJoCo SB3 Flat Walking，seed 42 | `c083476f1e4738e7d34d522fbb3b51be2b14a204` | `full/checkpoints/step_001179648000`，6000 次更新 | 50000 | 6084 |
| Newton RSL StandUp，seed 42 | `16950745457d0af017a4f2d68f1fbd079f8fa2ac` | `model_5000.pt`，实际已完成 5001 次更新 | 15000 | 5134 |
| Newton RSL StandUp，seed 43 | `16950745457d0af017a4f2d68f1fbd079f8fa2ac` | `model_5000.pt`，实际已完成 5001 次更新 | 15000 | 5094 |

Walking checkpoint 的 `model.zip` SHA-256 为 `588ac75930b2bb1318ce8c7c5613c49e925d457b576adf6ce0535ff5f518c2de`，`vecnormalize.pkl` SHA-256 为 `92de0eb867c36514e74c54c72aa794429fb3a0c3f7061977a1700fd5dedda604`。两个 StandUp checkpoint 的 SHA-256 分别为 `e5c8cea40a8d5876cf5418b1018213d72a522697fc739625b4a041f7d16a46e0` 和 `aced02d2dee2b1ce1ada89974238d5621d0e80b3b71c416f67d594e6c92203d2`。原始目录为服务器 `/mnt/data/users/jixin/workspace/code/Oh-My-Duck/outputs/experiments/jd-reproduction-20260923-04` 与同项目 `logs/rsl_rl/microduck_stand`。

RSL checkpoint 中的 `iter=5000` 是最后一次更新的编号；`infos.env_state.common_step_counter=120024` 对应 5001 次、每次 24 个控制步。接续训练需要将下一次更新编号设为 5001，执行剩余 9999 次更新，并保持原生 optimizer、观测归一化状态和课程进度。

三组训练的后台预览均未产生行为视频。预览进程寻找训练目录顶层的 `campaign.json`，独立 worker 的输出目录当时没有该文件。训练产物保留，预览修复已包含在 `f3a3cd2b69f10195ff060e8b9322babc8e3b4da8`。

## 数值与行为

读取 TensorBoard 的全部 scalar 标签并核查有限值：Walking 64 项，StandUp 每组 51 项。Walking 最后 5% 更新的 `train/approx_kl` 平均 0.0098，`train/explained_variance` 平均 0.8875，`train/value_loss` 平均 0.0792，`train/std` 平均 0.1472。StandUp seed 42/43 的最后 5% 更新平均 reward 分别为 40.01/34.60，episode length 均达到 300，`Episode_Termination/nan_state` 全程为 0；最后 5% 的 value loss 分别为 0.0618/0.0548，mean action std 分别为 0.1849/0.1756。课程设置在训练中变化，这些 reward 数值仅反映对应阶段的训练记录。

Walking 的原生 SB3 `VecNormalize` 已嵌入 ONNX，32 个样本的最大绝对数值误差为 `9.54e-7`。StandUp 两个 RSL checkpoint 的标量和 9 项惩罚符号检查通过；原生周期 ONNX 与官方导出路径各比较 65 个样本，最大绝对误差均为 0。

Walking 在 MuJoCo、评估 seed 42 下完成 700 个控制步，标准推扰与无推扰均未通过行为评分。无推扰时，前进命令 0.1 m/s 的实际平均前进速度为 0.0000305 m/s，转向命令 0.5 rad/s 的实际平均转向速度为 0.03450 rad/s；三个静止阶段通过。标准推扰时，对应速度为 0.02372 m/s 和 0.46535 rad/s，但运动与静止阶段的滑动窗口仍未通过。两种设置中机器人均完成全部控制步。

StandUp 在原生 Newton、无推扰、评估 seed 42 下，两份 checkpoint 均完成四个姿态各 400 个控制步。站立和坐姿通过，俯卧和仰卧未通过。seed 42 的俯卧/仰卧末端倾斜分别为 0.6003/0.5981 rad；seed 43 对应 0.5881/1.1097 rad，仰卧末端高度为 0.0592 m。当前结果仅覆盖一个评估 seed 与训练中途 checkpoint。

诊断的原始 `result.json`、轨迹、ONNX 与 TensorBoard 摘要位于服务器 `outputs/diagnostics/jd-interrupted-20260926`。Walking SB3 保留中途 checkpoint。新的原生 MuJoCo/RSL Walking 对照使用源码提交 `4253ee7901458b99903bb2bf7416ac727347b0f6`，已通过 64 环境启动、原生导出、真实恢复、8192 环境容量与导出检查，在 GPU6 执行 50000 次更新的完整训练。其 W&B online run ID 为 `4uezg0gj`，训练日志已记录初始更新，运行目录为 `outputs/experiments/jd-walking-rsl-control-20260926/mujoco-rsl-rl-walking-official-seed42`。

原生 RSL 恢复计数修复包含在提交 `eb8eb0ec5cbae6bb4b297dd271cd92373aa83555`。历史重复标签 checkpoint 在真实 Newton GPU 训练中连续恢复两次：保存标签 4、已完成 6 次、计数 144；随后两次保存标签 6/7、已完成 7/8 次、计数 168/192，optimizer step 从 120 增至 140、160。两组正式 StandUp 使用此提交的独立源码副本、8192 环境和 `9999` 次剩余更新预算，分别在 GPU2（seed 42，PID 535324，W&B online run `ghcs61i8`）及 GPU1（seed 43，PID 535196，W&B online run `xwhqqk1j`）启动。各自 `result.json` 的恢复校验记录原始计数 120024、已完成 5001 次、下一编号 5001；两份 `full.log` 均连续记录从 5001 到 5005 的首批更新。运行中的新环境计数需要等待下一次原生 checkpoint 保存才能直接读取。运行目录为 `outputs/experiments/jd-resume-20260926`。完整预算后的行为结论以最终评估为准。

修复后的独立预览命令已对 seed 42 的真实 `model_5000.pt` 完成 RSL 导出和 Newton 四姿态评估。`previews.json` 记录原始 checkpoint SHA-256、导出退出代码 0、行为评估退出代码 2。四份视频均为 1280×720、25 fps、8 秒，文件大小分别为站立 326002、坐姿 438154、俯卧 281624、仰卧 214806 字节；目录为服务器 `outputs/diagnostics/jd-interrupted-20260926/stand42-preview`。该预览检查视频生成流程，行为结果仍为 2/4。

## 官方资料与运行环境

针对 `pollen-robotics/mjlab_microduck/441tzs6d` 与 `pollen-robotics/mjlab_microduck/69u48n8l` 的 W&B API 读取均返回 403，当前账号无法取得其配置或 checkpoint。已经核对的源码范围为 Flat Walking 直接引用的 22 项 MDP 函数及 StandUp 的 31 项同源函数；此前在真实 64 环境中完成 `set_random_ground_state` 的 15 种配置下 `qpos`、`qvel`、随机数状态对照，见 [官方基线审查](official-baseline-audit-2026-09-08.md)。该检查不说明整套 Newton 物理行为与官方实现相同。

服务器系统 `libEGL.so.1.1.0` 文件大小为 0。项目已有的 `.cache/render-libs/osmesa/usr/lib/x86_64-linux-gnu/libEGL.so.1.1.0` 大小为 72352 字节。给 worker 指定该目录的 `LD_LIBRARY_PATH` 后，原生 `omd.py export` 成功完成同一 Walking checkpoint 的归一化导出检查。三组新训练的 `outputs/launches` 启动记录保存该环境路径、GPU、源码提交、进程号、节点 boot ID 与 cgroup 内存计数。
