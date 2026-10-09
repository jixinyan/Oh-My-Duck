# GPU 状态读取与资源检查

源码版本：`6bb4303e3b0772b0974834a60457855fa75b5d8e`。

`infrastructure/gpu_inventory.py` 提供原生查询参数、不可变设备与进程类型、CSV 读取及
独占设备检查。发布流程、训练测量和已有准备记录的接续使用相同的读取模块。
所有 CSV 内容由 Python 标准库读取。模块导入和设备状态查询不会初始化 CUDA。

## 实际记录测试

`tests/records/gpu-inventory-20261009/` 保存 `jd_B300` 于
`2026-10-09T16:41:28.871397+00:00` 返回的八张设备及三个 compute 进程记录。
三个进程记录包含两个唯一 PID。原始 CSV、查询参数与 SHA256 保存在 `capture.json`。

`tests/test_gpu_inventory.py` 的 31 项检查通过，覆盖原始身份和数量、重复进程行、
占用设备拒绝、可用显存、唯一设备身份、缺少字段及有限数值要求。
不完整或无效输入在调用位置抛出错误。测试中的无效 CSV 使用 `csv.writer` 生成。

测试输出：`outputs/acceptance/gpu-inventory-tests-20261009-02.xml`。

## 远程状态与持续空闲检查

实际发布流程的 `gpu_snapshot` 对 GPU 2 保存三次原生设备及进程采样，
通过十秒持续零 utilization、没有 compute PID 与至少 16 GiB 可用显存要求。
整个查询用时 26.11 秒。

GPU 4 的 utilization 为零，记录中仍有 PID `628870`、`631111`，独占检查拒绝接纳。
设备查询结果、进程 CSV 和每个文件的 SHA256 保存在
`outputs/acceptance/gpu-inventory-live-20261009-01/result.json`。
这是当时的状态记录；后续分配必须重新查询实际状态。

## 独立安装

Mac 从固定版本构建 wheel 和 source distribution，通过 439 个 source/resource 文件、
三份许可证与 29 项 CLI 调用；原生语音与 Harness 客户端文件 SHA256 一致。
记录位于 `outputs/acceptance/gpu-inventory-20261009-01-install/result.json` 和
同目录的 `independent-package.json`。

Ubuntu 22.04 独立环境通过相同 439 个文件、三份许可证和 27 项 CLI 调用。
安装后的 `GpuSampler` 查询实际设备与进程，拒绝 GPU 4 上已有的两个 PID，
完成 GPU 2 的显存采样并正常结束线程，记录中没有测量错误或其他会话的进程。
源码副本固定且干净，检查过程中没有导入 `torch`、`warp`、`isaaclab` 或 `isaacsim`。
记录位于 `outputs/acceptance/gpu-inventory-linux-20261009-01/package-audit.json`
及同目录的 `sampler/result.json`；原始 CSV、SHA256 与构建日志同时保存。

独立交付检查重新读取测试 XML、两个安装报告及所有原始 CSV，验证发布采样、
测量线程结束、四个改动模块的 Git/安装文件 SHA256、实际 `omd status` 输出和
保留的资产依赖文件，结果见 `outputs/acceptance/gpu-inventory-handoff-20261009-01.json`。

## 执行范围

本次检查查询实际 NVIDIA 状态并验证 CPU 代码，不启动训练、CUDA worker 或仿真。
当前 GPU 运行、RL 学习行为与 Microduck 真机继续保留独立验收要求。
GPU 验收与 RL 保持停止。资源模块的职责见
[源码说明](../../src/oh_my_duck/infrastructure/README.md)。
