# SPDX-License-Identifier: Apache-2.0
# Transform correction adapted from kabilankb/isaaclab-microduck at 4310fe0.
# See docs/third_party/isaaclab-microduck-LICENSE.txt and THIRD_PARTY_NOTICES.md.
"""Isaac Sim-only MJCF conversion worker. No physics training occurs here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from .paths import asset_dir, asset_source, project_root, source_fingerprint, model_file, ROBOT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(ROBOT_MODELS), default="walk")
    args = parser.parse_args()
    filename = model_file(args.model)
    stem = Path(filename).stem
    target = asset_dir(args.model)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Incomplete asset output exists: {target}; inspect it before rebuilding")
    # A private build directory prevents stale converter suffixes and half-built assets.
    with tempfile.TemporaryDirectory(prefix="build-", dir=target.parent) as tmp:
        work = Path(tmp)
        subprocess.run([str(project_root() / ".envs/mujoco/bin/python"), "-m", "omd_isaac.asset_reference",
            str(asset_source() / filename), str(work / "reference.json")], check=True)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        try:
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
            from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
            cfg = MJCFImporterConfig(mjcf_path=str(asset_source() / filename), usd_path=str(work),
                import_scene=False, merge_mesh=False, collision_from_visuals=False,
                collision_type="Convex Hull", allow_self_collision=True, fix_base=False)
            produced = Path(MJCFImporter(cfg).import_mjcf()).resolve()
            expected = work / stem / (stem + ".usda")
            if produced != expected.resolve() or not expected.is_file():
                raise RuntimeError(f"Converter output differs from expected path: {produced}")
            # Preserve original world transforms while removing ancestor rigid-body
            # transform inheritance. Reference: pinned kabilankb asset-conversion review.
            from pxr import Usd, UsdGeom, UsdPhysics
            stage = Usd.Stage.Open(str(expected))
            cache = UsdGeom.XformCache()
            fixes = []
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                ancestor = prim.GetParent()
                while ancestor and not ancestor.IsPseudoRoot():
                    if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                        fixes.append((prim.GetPath(), cache.GetLocalToWorldTransform(prim)))
                        break
                    ancestor = ancestor.GetParent()
            with Usd.EditContext(stage, stage.GetRootLayer()):
                for path, transform in fixes:
                    xform = UsdGeom.Xformable(stage.OverridePrim(path))
                    xform.ClearXformOpOrder()
                    op = xform.AddTransformOp()
                    op.Set(transform)
                    xform.SetXformOpOrder([op], resetXformStack=True)
            stage.GetRootLayer().Save()
            files = {str(p.relative_to(work)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(work.rglob("*")) if p.is_file()}
            (work / "build.json").write_text(json.dumps({"model": args.model, "source_fingerprint": source_fingerprint(args.model),
                "files": files, "transform_fixes": len(fixes), "status": "converted_not_physics_validated"}, indent=2) + "\n")
            work.rename(target)
        finally:
            app.close()

if __name__ == "__main__":
    main()
