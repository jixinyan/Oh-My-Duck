# 感知工具与 VLN 任务

`microduck.inspect_scene(prompt, source)` 在原生 Harness 的确认暂停状态读取当前 head RGB。返回值包括 episode、物理 sequence、目标框、目标表面距离、身体到目标的水平距离、bearing 和标注图像。Planner 在移动后重新调用工具，根据当前观测决定距离、角度和下一项操作。

`source="simulator_ground_truth"` 使用 Newton 当前相机的实例编号、实际形状名称和射线距离。目标来自当前可见像素；支持 `desk`、`chair`、`plant`、`bin`、`door`、`reception` 和 `objects`。每项结果明确记录 detection、mask 与 distance 的来源。

CPU MuJoCo/BAM 公寓使用相同工具，`perception/mujoco.py` 提供当前 RGB、depth、segmentation 和 world points。`objects` 返回当前可见的具名家具及互动目标；也可查询 `desk`、`chair`、`plant`、`table`、`cabinet`、`block`、`ball`、`dock` 等类别，或直接使用 `off_desk`、`obj_3` 等真实场景名称。未表示或不可见的目标返回空列表。五种实际状态、1253 个像素的原生几何求交、距离与角度、读取期间的物理状态保持及独立安装检查通过，见[CPU 感知验证](reports/cpu-scene-perception-2026-10-08.md)。

`source="models"` 调用独立 YOLO26/SAM 3.1 服务。YOLO26 提供 COCO 类别与目标框；启用 SAM 3.1 时，SAM 使用文本提示产生分割，服务通过框的 IoU 关联 YOLO 结果。SAM 的开放词汇结果可以没有 YOLO 类别匹配。服务的两种运行方式由启动参数明确选择，模型错误直接传播给调用者。

## 距离计算

距离需要与 RGB 对应的 calibrated depth geometry。Newton 接口从原生 ray buffer 获取每个像素的起点、方向和实际 ray-hit distance，再使用当前光心与相机旋转得到世界坐标。负值和非有限深度视为无效数据。

CPU 接口读取 MuJoCo 的 optical-axis depth，并使用实际单目 render frustum、光心和朝向计算世界坐标。背景与 clipping plane 之外的像素使用无效 depth。RGB、depth 与 segmentation 使用相同视图和从上到下的像素中心。感知读取保持 episode、物理 sequence、仿真时间和动作计数；后续相机读取继续返回 RGB。

目标表面距离取有效像素到光心距离的中位数，同时报告 10%–90% 区间。目标表面世界位置取各坐标的中位数；水平距离与 bearing 使用当前身体位置和 yaw。有效像素不足八个时返回 `insufficient_valid_depth`。正 bearing 表示向左。

YOLO 单独运行时使用框区域，结果可能包含背景深度；`mask_source="yolo26_bbox"` 保留这一含义。SAM 分割或 simulator ground truth 使用目标像素。RGB 单独输入不具备该模块要求的 calibrated depth geometry。真机适配需要提供对应的深度数据、相机标定与位姿；当前真机感知尚未验收。

## 独立服务

Linux NVIDIA 环境使用 `environments/perception/uv.lock`，与 Newton 环境分开。YOLO26 使用官方 headless Ultralytics 包；SAM 源码固定为 `2345a4ad109ac29c569da749c91d84f10dc08c40`。

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.envs/perception" uv sync --project environments/perception --locked
PYTHONPATH=src .envs/perception/bin/python scripts/fetch_perception_models.py --output checkpoints/perception
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .envs/perception/bin/python -m oh_my_duck.perception.service \
  --yolo checkpoints/perception/yolo26s.pt --output outputs/perception-service --port 8784
```

启用 SAM 3.1 需要已经获得官方模型访问权限的 Hugging Face 账号：

```sh
.envs/perception/bin/hf auth login
PYTHONPATH=src .envs/perception/bin/python scripts/fetch_perception_models.py \
  --output checkpoints/perception --include-sam
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 .envs/perception/bin/python -m oh_my_duck.perception.service \
  --yolo checkpoints/perception/yolo26s.pt --sam checkpoints/perception/sam3.1/sam3.1_multiplex.pt \
  --output outputs/perception-service-sam31 --port 8784
```

服务只监听 `127.0.0.1`，仿真 worker 通过 scene config 的 `perception_endpoint` 调用。每项响应记录实际 checkpoint SHA256 与源 RGB SHA256，客户端检查 episode 和物理 sequence。

`jd_B300` 已有权重 `/home/jixin/workspace/checkpoints/sam3.1/sam3.1_multiplex.pt`，SHA256 为 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`。可以将该路径直接传给 `--sam`。锁定环境使用 `setuptools==80.9.0`，满足固定 SAM 源码的 `pkg_resources` 依赖；`sam31.load_predictor` 检查 checkpoint 参数、派生 RoPE buffer 与 multiplex `init_state` 参数，将公共 session 接口的参数传递给实际 multiplex 方法。

联合服务已经在实际 Newton Office RGBD 上执行 SAM3.1 与 YOLO26。朝向为零的出生位置图像返回四个墙面分割、桌子、地面和植物，各项具有有效射线距离；另一朝向对同组提示返回空目标。YOLO 对部分分割的关联类别包含 `airplane` 和 `bench`。这些结果证明实际模型推理与距离传输可运行，识别准确率需要带有目标标注的独立评估。工具保留原始关联结果和空目标。

## 多阶段导航

`configs/simulation-demo/office-vln.json` 与对应 instruction 定义大厅中心、右侧绕行和办公桌接近任务。独立 Verifier 要求按顺序通过两个 checkpoint 区域，再进入最终目标区域并保持直立。物理访问记录包含实际位置与 sequence。

任务允许 simulator ground truth 的目标分割、距离、bearing 与 world odometry。它展示语言任务、感知工具、距离/角度 policy tools 和原生 Harness 的闭环；图像输入独立 VLN policy 与跨场景泛化需要各自的行为检查。

录制工具将 `inspect_scene` 的原生 image attachment 原样保存到所选 data directory 的 `tool-images/`。导出时指定 `--data-directory`，检查原始 attachment SHA256 和字节数。MP4 显示 observer 画面、带标注的 head RGB、实际工具参数与反馈、Planner 公开文字和独立 verdict。

```sh
python scripts/record_harness_demo.py export --base-url http://127.0.0.1:4348 \
  --run-id "$RUN_ID" --output ".cache/vln/$RUN_ID" --data-directory .cache/harness-perception-vln-20260930-03
python scripts/accept_harness_replay.py ".cache/vln/$RUN_ID" \
  --expected-verdict passed --expected-stop-reason policy_stop --require-stop-progress
python scripts/accept_vln_replay.py ".cache/vln/$RUN_ID" --scene-config configs/simulation-demo/office-vln.json
python scripts/render_microduck_run.py --export ".cache/vln/$RUN_ID" \
  --output outputs/demos/office-vln.mp4 --wall-speed 8
```

所有验证使用实际模型、Newton 相机、官方 policy、原生 ActionGate 和独立 Verifier。RL 保持停止。

VLN 检查保留每个 metric motion 的实际终态与误差，并检查多个模型行走/转向调用、至少三次不同物理帧的办公桌观测、按顺序通过 checkpoint 和最终正式 verdict。停靠距离使用实际身体位置到当前可见办公桌的 authored static USD geometry bounds；感知工具另行报告可见像素的表面距离中位数。`--require-metric-tools` 用于距离工具的严格精度验收，需要实际完成的行走目标；该项验收与导航路线完成分别记录。
