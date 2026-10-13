# 原生运动等待验证

验证日期：2026-10-13。

## 接口与来源

`microduck.wait_for_motion(max_wait_s)` 接受 1–90 秒整数窗口，默认 90。
调用返回当前物理进度与 `motion_wait.boundary_confirmed`。原生动作使用
EDH ActionGate、官方 ONNX 与 BAM 执行；完成的 policy Task 与清理 Task 传播执行错误。
Agent 根据实际执行状态继续监测，取得确认暂停边界后准备下一段动作。

| 内容 | 固定来源 |
| --- | --- |
| OMD 执行与安装源码 | `09eb8f7066ae93103164e3983e959c76a863f1ab` |
| EDH | `8a5e685b22d032207f53db20454f0992a4ad60fd` |
| 官方 policy | `1b56c396825c052a4e26e95cf2b8d8298af9e9b4` |
| 实际 policy | `velstand`、`alpha_walking` |
| 仿真 | CPU MuJoCo、公寓、XL330 M6 BAM、50 Hz 与四个物理子步 |
| renderer | OSMesa、llvmpipe LLVM 15.0.7 |
| seed | `20260929` |
| 出生位置 | `(-1.6, -0.5, π)` |
| 可见目标 | `ball_0`，实际相机与仿真 RGBD |

## 实际运行与独立复核

| 项目 | 结果 |
| --- | --- |
| 实际控制与物理子步 | 461 次、1,844 次 |
| 逐动作 ONNX 重新计算 | 461 次，最大误差 `0.0` |
| 原生动作发布记录 | 467 条，覆盖全部实际控制 |
| 真实等待结果 | 11 条 |
| 1 秒窗口结束后的进度 | sequence `2`、`4`、`77`，原生动作继续执行 |
| 90 秒窗口结束后的进度 | sequence `233`、`370`，`boundary_confirmed=false` |
| 90 秒工具调用完整耗时 | `90.3914566`、`90.3991435` 秒，包含随后读取传感器 |
| 取消等待调用 | policy Task 保持运行，后续实际 sequence 增加 |
| 确认暂停与终止后的重复读取 | sequence `75`、`411`、`461`，物理与停止计数保持一致 |
| 非法参数拒绝 | 七项，包括 boolean、越界整数、float、字符串与未知字段 |
| 目标接近 | 目标表面距离 `0.4240209885` 米；请求 `0.4` 米；误差 `0.0240209885` 米，标准 `0.05` 米 |
| 实际停止保持 | 50 个零 twist 控制步，最终连续停止样本 55 |
| 原生目标保持 | `goal_check.complete=true`，88 次保持，要求五次 |
| 接触与姿态 | 外部障碍接触 0，`fallen=false` |
| 终止 | `ended`、`device_confirmed=true`、`policy_stop` |
| 相机 | 92 张原始 640×480 PNG，全部解码与非均匀像素检查通过 |
| 资源 | session 释放；所属 PID `258937` 退出状态为零；控制端口关闭 |
| CUDA | visibility 为空、初始化状态为 false |

保存的原始动作包含 61 维观测、14 维动作与 13 维命令。
独立复核使用固定官方 ONNX 重新计算全部动作，并核对其物理 sequence、
原生发布记录、身体位置和等待结果。图像检查使用原始 PNG 字节与 SHA256。

## 安装与公开入口

Mac 和 Linux 分别构建并安装实际 wheel 与源码包，在仓库外运行公开命令。
两个主机各通过 455 个源码及资源文件、三份 license 和 34 项 CLI 调用。
依赖检查通过；100 项场景、导入与等待参数检查通过，Node 入口语法检查通过。

新增公开验证入口：

```sh
CUDA_VISIBLE_DEVICES='' python -m oh_my_duck.infrastructure.headless \
  --mujoco-renderer osmesa -- python -m oh_my_duck validate motion-wait \
  --catalog "$POLICY_DIRECTORY" --edh-source "$HARNESS_DIRECTORY" \
  --output outputs/acceptance/motion-wait-new
```

工具使用与源码位置见[运动等待](../native-motion-wait.md)。

## 原始资料

| 内容 | 路径 |
| --- | --- |
| 远程完整事件、动作、测量、工具、PNG 与资源记录 | `/home/jixin/workspace/code/Oh-My-Duck/outputs/acceptance/motion-wait-linux-20261013-01/` |
| 本机原始结果、工具与复核记录 | `outputs/acceptance/motion-wait-linux-20261013-01/` |
| Mac 安装复核 | `outputs/acceptance/motion-wait-package-mac-20261013-01/result.json` |
| Linux 安装复核 | `outputs/acceptance/motion-wait-package-linux-20261013-02.json` |
| 场景、导入与等待参数检查 | `outputs/acceptance/motion-wait-imports-20261013-03.xml` |
| 验证汇总与来源检查 | `outputs/acceptance/motion-wait-handoff-20261013-01.json` |

当前证据覆盖实际 CPU 原生工具和物理动作。该执行使用直接原生控制接口；
完整模型任务与独立 Verifier 的来源见[模型目标任务](model-target-task-2026-10-13.md)。
当前等待参数的完整模型任务、Newton GPU 与真机执行需要各自验证。
GPU 验收和 RL 保持停止。
