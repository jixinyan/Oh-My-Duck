# SSH 空闲连接、CPU 服务与资源关闭

日期：2026-10-13。SSH 与安装检查源码：
`932b75c6d5393a6b24704d3ea8f116b0d4fe9b52`。
远程 CPU perception 服务源码：`7fdaf000d1851e8c981d8cb46a06418d4fb379a9`。

## 共享连接参数

`src/oh_my_duck/infrastructure/ssh.py` 的 `ssh_command` 供 provider 读取、
发布验收和语音检查使用。host、command、keepalive count 和 forwarded ports
在建立连接前验证。连接使用十秒建立期限、三十秒 keepalive 间隔、
默认三次无回应期限和 `ExitOnForwardFailure=yes`。发布状态查询使用一次无回应期限。
本地转发绑定 `127.0.0.1`。Node 原生 worker 使用三十秒间隔与三次无回应期限。

## 实际连接与关闭

Mac 通过共享参数连接 `jd_B300`，使用新端口转发到真实 CPU YOLO26 服务。
原始 `ssh -G` 输出确认有效配置。服务完成权重加载，实际 parameter device 为 `cpu`，
CUDA 尚未初始化。检查在初始 HTTP health 成功之后保持 660 秒应用空闲。
期间仅检查本次 SSH 进程，保存 67 项连续状态记录。

| 项目 | 实际结果 |
| --- | --- |
| 应用空闲时长 | 660.002102958 秒 |
| 空闲期间应用请求 | 0 |
| 初始与结束 HTTP health | 相同；模型 SHA256、CPU 与 CUDA 状态一致 |
| SSH 状态采样 | 67 项，连续运行 |
| 结束操作 | 关闭所属 stdin 控制连接，远程 supervisor 关闭所属服务 |
| SSH / 服务退出 | 0 / 0，无 graceful cleanup 超时 |
| 两端端口 | 本地转发与远程服务端口关闭 |
| CPU worker | 数量为 0 |

检查使用实际 OpenSSH、真实远程服务和原生进程退出记录。
GPU 验收与 RL 保持停止，其他用户的进程保持原有状态。

## 安装与原始任务复核

Mac 和 Linux 各自在没有 tracked 改动的固定源码目录构建 wheel 与 source distribution，
在 checkout 之外执行安装版 CLI：457 个 source/resource 文件、三份许可证、
34 项命令与实际依赖检查通过。121 项参数、场景、等待和导入检查通过，
其中包含十九项 SSH 非法参数拒绝。

新安装包使用原始 Luna 任务的真实事件与相机资料，严格复核停止进度、目标跟踪、
模型来源和运动等待。结果与该任务的独立检查一致：485 次控制、1,940 次物理子步、
十条等待、正式 Verifier `passed`。该项是原始任务复核；实际任务来源见
[完整模型等待任务](model-motion-wait-2026-10-13.md)。

## 资料

以下路径位于 `outputs/acceptance/`：

| 内容 | 路径 |
| --- | --- |
| 独立检查 | `ssh-idle-handoff-20261013-01.json` |
| 实际连接、有效配置、67 项采样和退出 | `ssh-idle-mac-20261013-01/` |
| 远程服务与所属进程记录 | `ssh-idle-perception-linux-20261013-01/` |
| Mac 安装 | `ssh-idle-package-mac-20261013-01/result.json` |
| Linux 安装 | `ssh-idle-package-linux-20261013-01.json` |
| 原始任务的安装版回放 | `ssh-idle-installed-replay-mac-20261013-02.json` |
| 参数与导入检查 | `ssh-idle-imports-20261013-01.xml` |

源码位置见[Architecture](../architecture.md)，使用参数见
[原生部署](../harness-native-integration.md#ssh-连接与远程进程)。
完整模型任务服务管理、网络中断统计、GPU 与真机分别验收。
用户原有 `environments/isaac-assets/uv.lock` 保持原有内容。
