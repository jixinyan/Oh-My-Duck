# 原生 Harness 准备、模型调用与 CPU 执行

固定实现源码：`bcbc2c32a33e8446d23cf6c9f647d0e28b5a14f6`。

`omd setup-harness` 准备固定原生 SDK、Node 依赖、官方策略和锁定 CPU 环境。
已有 EDH Git 仓库通过 `--edh-repository` 指定，准备记录通过 `--output`
写入新的目录。准备流程保存八项输入的 SHA256、原始下载、每个阶段的日志和
所属进程退出记录；错误在对应位置传播，已有输出目录拒绝再次使用。

## 原生 Node SDK 与官方策略

Mac 的独立安装通过公开命令创建固定 EDH 源码，使用 `pnpm@11.19.0` 和
`--frozen-lockfile` 安装依赖。六个阶段正常退出，跟踪源码保持干净。
实际 Node 版本为 `25.4.0`，五项原生 SDK 导出和 physical Schema 加载通过。

EDH 来源为 `8a5e685b22d032207f53db20454f0992a4ad60fd`。
官方策略来源为 `pollen-robotics/microduck-policies` 的
`1b56c396825c052a4e26e95cf2b8d8298af9e9b4`。
manifest 和十个 ONNX 均通过实际下载、长度及固定 SHA256 检查，原始下载保存完整。
独立复核重新比较 Git 输入、固定 SDK、原始文件、目标文件和阶段进程状态。

证据：`outputs/acceptance/native-preparation-mac-20261011-02/`
中的 `result.json`、`independent.json`、`downloads/` 和各阶段记录。

## 实际模型调用

Mac 与 Linux 的独立安装均通过新准备的原生 SDK 调用 `gpt-6-luna`。
两次请求实际使用 `responses`、`high` 和 8192 token，记录包含 HTTP 200、
`response.completed` 和 `finish.kind=stop`。两个所属进程正常退出。
凭据从用户指定的私有配置读取，通过进程环境传递。

证据：`outputs/acceptance/native-preparation-model-mac-20261011-01/`
与 `outputs/acceptance/native-preparation-model-linux-20261011-01/`
中的 `result.json`、`process.json`、`stdout.json` 和 `data/model-transport.jsonl`。
模型连接检查没有创建机器人会话。

## 接口与独立安装

准备入口、共享模型设置、任务参数、发布计划、实际 GPU 记录和延迟导入检查
通过 140 项测试。原生 Python SDK 检查接入后，23 项模块导入检查通过。
实际错误 Git 来源和重复输出目录均返回退出状态 1，三个所属进程结束，
原始失败记录的 SHA256 保持一致。

Mac 与 Ubuntu 独立构建安装通过 443 个源码及资源文件、三份许可证。
项目目录之外的公开命令检查分别通过 30 项和 28 项。
Mac 使用新下载的官方策略复核保存的 1,208 次实际输入输出，ONNX 最大误差为零；
对应五项运动及 4,832 次物理步骤的保存资料复核通过。
这些资料保留原始执行的源码和时间。

证据：

- `outputs/acceptance/harness-preparation-tests-20261011-01.xml`
- `outputs/acceptance/harness-sdk-imports-20261011-01.xml`
- `outputs/acceptance/native-preparation-rejections-20261011-01/result.json`
- `outputs/acceptance/native-preparation-20261011-02-install/downloaded-policies.json`
- `outputs/acceptance/native-preparation-linux-20261011-01/package-audit-client.json`

## 锁定 CPU 环境与连续物理运行

完整准备流程使用 `environments/cpu-apartment/uv.lock`，安装固定原生 Python SDK，
并通过 `omd doctor --runtime cpu-apartment` 检查实际依赖、CPU Torch、MuJoCo、
十个 ONNX 和 Python SDK 文件。SDK 检查逐个比较安装文件与固定源码。

Linux 的新环境通过公开准备入口完成九个阶段，全部所属进程正常退出。
实际依赖为 CPU Torch `2.9.1+cpu`、MuJoCo `3.8.0`、mjlab `1.3.0`、
BAM `1.0.1`、Warp `1.12.0` 和 ONNX Runtime `1.30.0`。
28 个原生 Python SDK 文件逐个通过固定源码 SHA256 检查；十个策略实际执行
CPU ONNX 推理。Torch 的 CUDA 版本为 `None`，整个流程没有初始化 CUDA。

准备使用与锁定文件一致的 Torch 和 Warp wheel，分别核查固定 SHA256 后通过
UV 安装，完整 `uv sync --locked` 核查所有依赖。环境路径为
`.envs/native-preparation-cpu-20261011-02`，SDK 路径为
`.cache/edh-native-preparation-linux-20261011-02`。

新环境通过 `omd validate task-continuation` 执行两个直接任务和两个原生
worker 任务，共 300 次控制、1,200 次物理步骤。连续任务保持当前物理状态、
episode 与 sequence，每项任务保存独立的起始状态和目标范围。
只读检查保持物理时间，执行结束具有设备确认，所有会话和所属进程正常关闭。

独立复核重新计算 150 次实际的 61→14 ONNX 动作，最大误差为零；
核对实际 HOME、动作缩放、controller target 和每次四个物理步骤。
150 张原始 head RGB PNG 完整解码通过，均为 `320×240`，像素具有有效内容；
图像统计、采集时间和 `8×8` ToF 数据均与实际记录一致。
本项检查没有执行模型导航任务；此前模型调用和任务验收使用各自的原始记录。
GPU 验收与 RL 保持停止；Newton 运行和 Microduck 设备保留独立验收要求。

证据：

- `outputs/acceptance/native-preparation-linux-setup-20261011-02/result.json`
- `outputs/acceptance/native-preparation-linux-setup-20261011-02/independent.json`
- `outputs/acceptance/native-preparation-continuation-linux-20261011-02/`
- `outputs/acceptance/native-preparation-continuation-runner-linux-20261011-02/`

入口交付复核见 `outputs/acceptance/native-preparation-entry-handoff-20261011-02.json`，
核查两个模型响应、安装文件、命令结束、测试 XML、实际接纳拒绝、保存资料和
用户修改的资产依赖文件。完整项目范围继续见[设计](../Agentic%20Microduck%20-%20Project%20Design%20v0.1.md)
和[执行计划](../Agentic%20Microduck%20-%20Execution%20Plan%20v0.1.md)。
