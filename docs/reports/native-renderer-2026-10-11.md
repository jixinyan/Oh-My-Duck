# 原生 MuJoCo renderer 与 CPU 执行

固定实现源码：`16d1fd574ea71d726f37884cfa5f054bba6dbdab`。

`omd harness --mujoco-renderer osmesa` 明确选择 Linux 无界面 CPU 渲染。
本地进程与 SSH worker 在导入 MuJoCo、PyOpenGL 之前接收 renderer 和动态库设置。
SSH worker 通过同一个 `oh_my_duck.infrastructure.headless` 入口执行。
参数接受 `egl` 与 `osmesa`；显式选择要求 CPU MuJoCo 场景。
原生 Harness、ActionGate、任务目标、执行预算、policy 和物理控制继续使用各自的固定实现。

Ubuntu 22.04 amd64 可以使用已有命令准备固定 OSMesa：

```sh
CUDA_VISIBLE_DEVICES='' python omd.py setup --backend mujoco --software-renderer --skip-env
CUDA_VISIBLE_DEVICES='' python omd.py harness --mujoco-renderer osmesa
```

模型配置、SDK、策略和仿真 Python 按[原生集成说明](../harness-native-integration.md)设置。
源码位置为 `cli/harness.py`、`infrastructure/headless.py` 和
`integrations/edh/server.mjs`。

## 公开参数和实际会话

五个实际 CLI 进程验证帮助信息、renderer 参数和子进程要求。
无效 renderer 返回退出状态 2；Newton 场景与 CPU renderer 的组合在读取
provider 或启动仿真之前返回退出状态 1；缺少命令返回 2，缺少 executable 返回 1。

清除继承的 renderer、PyOpenGL 和动态库设置之后，Linux 本地和 Mac→SSH
分别通过公开命令建立并关闭原生会话。会话资源释放，两个所属服务进程正常退出。
这两项检查仅验证实际启动与关闭。

证据：

- `outputs/acceptance/mujoco-renderer-arguments-20261011-01/`
- `outputs/acceptance/native-renderer-linux-20261011-01/`
- `outputs/acceptance/native-renderer-ssh-mac-20261011-01/`

## CPU 物理执行

实际 `headless` 入口在清除继承设置后启动
`omd validate task-continuation`。两个直接任务和两个原生 worker 任务共
300 次控制、1,200 次物理步骤，使用固定官方 ONNX、MuJoCo/BAM 和 llvmpipe。
原生记录包含两项独立任务的 150 次控制、600 次物理步骤、设备确认的
`policy_stop` 和资源释放。

独立复核在执行主机重新计算 150 次实际的 61→14 ONNX 动作，最大误差为零。
HOME、动作缩放、controller target、sequence 和每次四个物理步骤均通过核查。
150 张原始 head RGB PNG 解码通过，尺寸为 `320×240`，像素包含有效内容；
图像统计、采集时间与 `8×8` ToF 数据和原始记录一致。
CPU Torch 的 CUDA 版本为 `None`，验证期间没有初始化 CUDA。

证据：`outputs/acceptance/native-renderer-physics-20261011-01/`。

## 独立构建安装

Mac 与 Linux 从干净的固定 Git 源码构建 wheel 和 source distribution，
分别安装到独立环境。每台主机均通过 444 个源码及资源文件、三份许可证、
项目目录之外的 28 项实际 CLI 检查。112 项命令开始与退出记录完整，
全部退出状态为零；两个平台的构建产物分别保存 SHA256。

证据：

- `outputs/acceptance/native-renderer-mac-20261011-01-install/`
- `outputs/acceptance/native-renderer-linux-package-20261011-01/`
- `outputs/packages/native-renderer-mac-20261011-01/`
- `outputs/packages/native-renderer-linux-20261011-01/`

交付复核为 `outputs/acceptance/native-renderer-handoff-20261011-01.json`，
保存三个实现文件、原始报告、会话、进程与物理资料的 SHA256。
完整模型导航、当前 Newton 运行和 Microduck 设备继续使用独立验收要求。
GPU 验收与 RL 保持停止。
