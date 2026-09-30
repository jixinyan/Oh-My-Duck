# Isaac 场景资源与来源

日期：2026-09-29。

场景目录由 `configs/simulation-scenes.json` 定义。`scripts/fetch_isaac_scenes.py` 获取官方 USD、全部目录资源、材质和纹理，并保存 `inventory.json` 与不可变的 `asset-provenance.json`。资源目录使用 `data/scenes/<scene_id>/source/`，保留上游相对引用。原始 USD 文件保持原样。

## 已取得的场景

| scene_id | 官方目录 | 几何检查 | 完整来源校验 |
|---|---:|---|---|
| `nvidia-office-6.0` | 2,290 个文件，740,501,164 bytes | 3,785 个 Mesh，3,645 个 `none` triangle collider，1 个启用碰撞的 Plane | 已通过；含外部 HDR 共 2,291 个文件 |
| `nvidia-hospital-6.0` | 1,638 个文件，1,097,745,765 bytes | 2,058 个 Mesh，2,035 个 `none` triangle collider，1 个 Plane | 待完成逐文件校验与来源清单 |
| `nvidia-warehouse-6.0` | 2,715 个文件，538,289,878 bytes | 待检查 | 已取得部分文件，待完成获取与校验 |

Office 的 `asset-provenance.json` 位于远程服务器的 `/home/jixin/workspace/code/Oh-My-Duck/data/scenes/nvidia-office-6.0/`。其依赖检查覆盖 543 个 USD layer 和 357 个 asset，没有未解析的 USD 资源。每个资源记录来源、ETag、LastModified、大小和 SHA256。目录外的 `sunflowers_4k.hdr` 同样记录来源与校验值。

原始 `office.usd` 的 SHA256 为 `32b83ba60d0b12d953ed693bb1c2ab8c51332defc4bbb8240eced5ad2e2ce8d0`。完整来源清单的 SHA256 为 `1093dcbb0a25799fef75bbdce34fd2f82ed7a70cd51cc5c339993fa0d484c681`。同一获取命令的第二次执行已通过全部 SHA256 复核，并保持来源清单不变。

下载请求使用 `If-Match` 固定 inventory 中的 ETag。已有文件通过之前的 SHA256 清单校验；首次检查已有文件时，单部分 S3 ETag 与文件 MD5 比较，multipart ETag 使用 `If-Match` 重新获取对应资源并比较完整内容。来源清单已经存在时，工具验证内容，保持已有清单不变。HTTP、解析、大小、来源或校验失败会终止当前执行。

USD 依赖通过 `UsdUtils.ComputeAllDependencies` 读取。场景使用的 `OmniPBR.mdl` 与 `OmniGlass.mdl` 属于 Isaac Sim 的标准 MDL library；若场景引用这些模块，来源清单记录运行时模块依赖。工具完整获取场景目录内的 MDL 文件与纹理。

## Office 几何与配置

Office 使用 Z-up、`metersPerUnit=1` 和 `/Root` defaultPrim。原始静态 Mesh 的碰撞近似为 `none`，地面提供 `/Root/GroundPlane/CollisionPlane`。

`geometry-audit.json` 记录原始 Mesh 的世界坐标边界与碰撞属性。`navigation-grid.json` 从原始 floor tile 与启用碰撞的 Mesh 边界产生保守通行网格，分辨率为 0.1 m，机器人身体余量为 0.25 m。`public-map.json` 提供对应的几何来源校验值、开阔区和附近物体边界。它们用于规划提示；物理碰撞仍由导入的原始场景计算。

所选开阔位置为 `(-17.5500013, 30.4499995)`，网格计算的净空为 4.9 m。`configs/simulation-demo/office.json` 使用该位置和官方初始 base height `0.125 m`，目标为同一开阔区域内沿 x 方向行走 0.5 m 并停止。配置包含 `backend`、`scene_id`、`usd_path`、`provenance_path`、`public_map_path`、`spawn_pose`、`goal` 和 `budget`。

Newton 的 MuJoCo solver 在原生 MuJoCo contacts 路径中将 Mesh 编译为 convex hull。室内门口及凹形家具需要保留实际碰撞几何。导入、求解器接触、机器人动作、传感器与停止行为的验收由实际 Isaac/Newton runtime 检查；本报告中的资源校验不代表物理行为已经验收。

## 上游与使用范围

官方场景目录与下载位置见 [Isaac Sim Environment Assets](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_environments.html)。资源版本固定为官方 `Assets/Isaac/6.0`，与仓库现有 Isaac Lab asset root 一致。

[NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/latest/common/license-isaac-sim-additional.html) 适用于相关 NVIDIA 模型和材质。资源用于 NVIDIA GPU 系统中的 Isaac Sim / Isaac Lab；下载资源保存在数据目录，代码提交只包含目录配置、获取工具与来源说明。公开分发代码时不得附带这些资源。

用户提到的 [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) 将 OmniGibson 代码与 dataset 使用许可分别规定。其官方 [setup.sh](https://github.com/StanfordVL/BEHAVIOR-1K/blob/main/setup.sh) 内的数据许可要求在 OmniGibson 内进行非商业学术研究，并限制提取与再分发。其官方 [Isaac Sim 运行说明](https://behavior.stanford.edu/omnigibson/under_the_hood.html) 使用 PhysX 作为物理状态来源。BEHAVIOR 的原生接入需要独立遵守这些数据与运行要求。

## 继续工作的位置

当前停止点为 Office 的真实 Isaac/Newton policy、传感器和停止行为验收。Hospital 与 Warehouse 的数据文件保持保存状态。后续完成各自来源清单、实际几何坐标、policy 演示配置与物理行为验收。
