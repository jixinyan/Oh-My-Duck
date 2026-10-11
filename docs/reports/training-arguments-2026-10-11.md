# 训练与评估参数验证 — 2026-10-11

执行源码：`5123a422d0b29b0aa22cf12cc4c8d2fe99db0c67`。
GPU 运行验收与 RL 训练保持停止。

## 参数与入口

`src/oh_my_duck/rl/training/arguments.py` 使用 `argparse` 读取唯一的
`--task` 选项。`--task NAME` 与 `--task=NAME` 使用相同任务选择。
缺少选项值或重复提供选项时，CLI 返回退出码 2。

MuJoCo 与 Isaac/Newton 的注册任务评估使用
`oh_my_duck.rl.evaluation.task`。训练任务使用第一个位置参数，
RSL-RL 与 SB3 使用各自的原生训练模块。Isaac/Newton 的 PD 诊断模式使用
明确的 `--task Omd-Microduck-PD-Diagnostic-v0`，支持两种选项写法。
运行名称、输出目录与各框架参数完整传递。

模块位置与职责见[RL 源码说明](../../src/oh_my_duck/rl/README.md)。

## 实际验证

| 检查 | 结果 |
| --- | --- |
| 参数选择与独立模块导入 | 27 项通过，零失败、错误或跳过 |
| 两种物理环境的评估命令生成 | 六项通过，包含两种任务选项写法与未提供任务选项的入口 |
| 原生训练命令生成 | 20 项通过，包含 Walking、StandUp、两种物理环境、RSL-RL、SB3 与明确的 PD 诊断选择 |
| Linux 安装后的公开评估 CLI | 八次调用结束；四次帮助调用返回 0，四次无效参数调用返回 2 |
| Mac 独立安装 | wheel 与 sdist 核对 444 个源码及资源文件、三份 license、30 项 CLI 调用 |
| Linux 独立安装 | wheel 与 sdist 核对 444 个源码及资源文件、三份 license、28 项 CLI 调用 |

Linux 命令使用已有的实际 MuJoCo、MuJoCo/SB3 与 Isaac/Newton Python 环境。
公开评估入口的帮助调用在载入执行依赖之前结束。
命令生成检查保留参数列表，核对原生模块与 logger 选项，未执行训练。
每次公开评估调用保存 stdout、stderr、进程编号和退出状态。

安装检查在工作目录外运行公开 CLI，并核对安装内容与固定源码。
Mac 的额外两项调用独立检查已有的 1,208 次 ONNX 控制记录与连续五项运动记录；
这些记录使用其原始执行源码，当前检查属于保存数据审查。

## 验证文件

- `outputs/acceptance/training-arguments-tests-20261011-03.xml`
- `outputs/acceptance/training-arguments-linux-20261011-01/routes/result.json`
- `outputs/acceptance/training-arguments-linux-20261011-01/routes/*.process.json`
- `outputs/acceptance/training-arguments-linux-20261011-01/package-audit.json`
- `outputs/acceptance/training-arguments-20261011-02-install/result.json`
- `outputs/acceptance/training-arguments-handoff-20261011-01.json`

独立检查核对源码 SHA、两台主机的包检查、每项 CLI 的开始和结束记录、
八次公开评估调用的日志与实际退出码，并核对保留的 Isaac 依赖锁文件。

## 运行验收

有效训练策略、Newton 当前控制器、长导航和 Microduck 设备保持各自的
验收要求。Linux 完整锁定 CPU 环境与新环境的连续物理运行状态见
[原生准备入口记录](native-preparation-2026-10-11.md)。
