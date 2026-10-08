import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

from oh_my_duck.experience.harness_replay import save_json
from oh_my_duck.infrastructure.usd_runtime import verify_usd_runtime


def mesh_snapshot(stage):
    import numpy as np
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    meshes = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            transform = np.asarray(cache.GetLocalToWorldTransform(prim))
            if not np.isfinite(points).all() or not np.isfinite(transform).all():
                raise ValueError(f"USD mesh 包含无效坐标: {prim.GetPath()}")
            meshes[str(prim.GetPath())] = (points, transform)
    if not meshes:
        raise ValueError("USD 资产没有 mesh")
    return meshes


def audit_model(model):
    import mujoco
    import newton
    import numpy as np
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    from oh_my_duck.robotics.microduck.protocol import JOINT_NAMES
    from oh_my_duck.rl.backends.isaac_newton.paths import require_asset
    from oh_my_duck.rl.backends.isaac_newton.task_binding.collision_assets import (
        prepare_collision_assets, reference_model, source_geom,
    )

    path = require_asset(model)
    directory = path.parent.parent
    manifest = json.loads((directory / "build.json").read_text())
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in manifest["files"]}
    session = Sdf.Layer.CreateAnonymous("cpu-asset-audit.usda")
    stage = Usd.Stage.Open(Sdf.Layer.FindOrOpen(str(path)), session)
    stage.SetEditTarget(session)
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"USD 资产缺少 default prim: {path}")
    before = mesh_snapshot(stage)
    reference = reference_model(model)
    prepared = prepare_collision_assets(root, reference)
    if any(prim.IsInstance() for prim in Usd.PrimRange(root)):
        raise AssertionError("碰撞准备之后仍然存在 USD instance")
    geoms = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        geom = source_geom(str(prim.GetPath()), reference)
        geoms.append(geom)
        material, relation = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        expected = root.GetPath().AppendChild("PhysicsMaterial" + str(geom))
        if not material or not relation or material.GetPath() != expected:
            raise AssertionError(f"碰撞材质绑定与源码几何不一致: {prim.GetPath()}")
        physics = UsdPhysics.MaterialAPI(material.GetPrim())
        np.testing.assert_allclose([physics.GetStaticFrictionAttr().Get(), physics.GetDynamicFrictionAttr().Get()],
                                   [reference.geom_friction[geom, 0]] * 2, atol=1e-7, rtol=0)
        if physics.GetRestitutionAttr().Get() != 0.0:
            raise AssertionError(f"碰撞材质 restitution 与官方配置不一致: {prim.GetPath()}")
        for attribute, value in (("newton:maxHullVertices", -1), ("mjc:maxhullvert", -1),
                                  ("mjc:condim", int(reference.geom_condim[geom])),
                                  ("mjc:priority", int(reference.geom_priority[geom]))):
            if prim.GetAttribute(attribute).Get() != value:
                raise AssertionError(f"USD 碰撞属性与源码不一致: {prim.GetPath()} {attribute}")
    expected = {index for index in range(reference.ngeom) if reference.geom_contype[index] or reference.geom_conaffinity[index]}
    if set(geoms) != expected or len(geoms) != len(set(geoms)) or geoms != prepared["collision_geoms"]:
        raise AssertionError("USD enabled collider 与官方模型的完整集合不一致")
    after = mesh_snapshot(stage)
    if set(before) != set(after):
        raise AssertionError("碰撞准备改变了 mesh 集合")
    for label, (points, transform) in before.items():
        np.testing.assert_array_equal(after[label][0], points)
        np.testing.assert_array_equal(after[label][1], transform)
    builder = newton.ModelBuilder()
    builder.add_usd(stage)
    imported = {Path(label).name for label in builder.joint_label}
    joints = {reference.joint(index).name for index in range(reference.njnt)
              if reference.jnt_type[index] != mujoco.mjtJoint.mjJNT_FREE}
    if not set(JOINT_NAMES).issubset(imported) or not joints.issubset(imported):
        raise AssertionError("Newton 导入缺少官方 servo 或 passive joint")
    for name, digest in hashes.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise AssertionError(f"资产检查改变了生成文件: {name}")
    return {"model": model, "asset": str(path), "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generated_files_sha256": hashes, "collision_geoms_verified": len(geoms),
            "instances_expanded": prepared["instances_expanded"], "meshes_verified": len(before),
            "mesh_points_verified": sum(len(points) for points, _ in before.values()),
            "friction_and_contact_attributes_verified": True, "physics_material_bindings_verified": True,
            "mesh_points_and_world_transforms_unchanged": True, "generated_files_unchanged": True,
            "newton_body_count": builder.body_count, "newton_shape_count": builder.shape_count,
            "newton_joint_count": builder.joint_count, "canonical_servos_imported": len(JOINT_NAMES),
            "source_nonfree_joints_verified": len(joints)}


def main():
    parser = argparse.ArgumentParser(description="使用 CPU 核验实际 USD 碰撞、官方材质和 Newton 导入")
    parser.add_argument("--models", nargs="+", choices=("allcollisions", "groundcontact_rollers"),
                        default=["allcollisions", "groundcontact_rollers"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or len(args.models) != len(set(args.models)):
        parser.error("repeat 必须为正整数，models 必须没有重复项")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU 资产验收要求明确设置 CUDA_VISIBLE_DEVICES 为空字符串")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "scope": "CPU USD preparation and native Newton import",
              "gpu_acceptance_performed": False, "checks": []}
    try:
        result["provider"] = verify_usd_runtime()
        import torch
        import warp as wp
        from pxr import Usd, Work
        from oh_my_duck.infrastructure.provenance import source_provenance

        if Usd.GetVersion() != (0, 26, 8) or torch.cuda.is_initialized() or any(device.is_cuda for device in wp.get_devices()):
            raise RuntimeError("资产验收需要 OpenUSD 26.08 和仅有 CPU 的执行环境")
        result.update(provenance=source_provenance(), usd_version=Usd.GetVersion(),
                      usd_worker_threads=Work.GetConcurrencyLimit(), available_warp_devices=[str(device) for device in wp.get_devices()],
                      packages={name: version(name) for name in ("newton", "mujoco", "warp-lang", "torch")})
        with (args.output / "progress.jsonl").open("x") as progress, wp.ScopedDevice("cpu"):
            for repetition in range(args.repeat):
                for model in args.models:
                    progress.write(json.dumps({"event": "started", "repetition": repetition + 1, "model": model}) + "\n")
                    progress.flush()
                    checked = {"repetition": repetition + 1, **audit_model(model)}
                    result["checks"].append(checked)
                    progress.write(json.dumps({"event": "verified", **checked}, allow_nan=False) + "\n")
                    progress.flush()
        if torch.cuda.is_initialized() or any(device.is_cuda for device in wp.get_devices()):
            raise RuntimeError("CPU 资产验收期间出现 CUDA 初始化")
        result.update(passed=True, torch_cuda_initialized=False, actual_newton_imports=len(result["checks"]))
    finally:
        if failure := sys.exception():
            result.update(error_type=type(failure).__name__, error=str(failure))
        save_json(args.output / "result.json", result)
    print(json.dumps({"passed": True, "actual_newton_imports": result["actual_newton_imports"],
                      "gpu_acceptance_performed": False}))


if __name__ == "__main__":
    main()
