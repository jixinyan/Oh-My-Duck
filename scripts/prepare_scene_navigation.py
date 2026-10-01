import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--spawn", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.scene_root / "asset-provenance.json").read_text())
    source = args.scene_root / "source" / manifest["source"]["usd_relative_path"]
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    grounds, obstacles = [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if prim.GetTypeName() == "Plane":
            grounds.append(str(prim.GetPath()))
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower, upper = list(bounds.GetMin()), list(bounds.GetMax())
        if lower[2] < 0.4 and upper[2] > 0.06:
            obstacles.append({"prim_path": str(prim.GetPath()), "min": lower, "max": upper})
    if not grounds or not obstacles:
        raise ValueError("Scene requires native ground and obstacle geometry")
    nearby = [item for item in obstacles if sum(max(item["min"][i] - args.spawn[i],
                args.spawn[i] - item["max"][i], 0) ** 2 for i in range(2)) < 64]
    if not nearby:
        raise ValueError("Spawn neighborhood has no scene obstacle geometry")
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({"scene_id": manifest["scene_id"], "frame_id": "world",
        "source_kind": "authored USD ground and conservative obstacle bounds",
        "source_usd_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "ground_collider_paths": grounds, "spawn_xy_m": args.spawn,
        "obstacles": nearby, "all_obstacle_count": len(obstacles), "map_radius_m": 8,
        "navigation_limits": "Bounds are planning information; native contacts remain authoritative"}, indent=2) + "\n")
    print(json.dumps({"map": str(output), "ground_paths": grounds, "obstacle_count": len(obstacles)}))


if __name__ == "__main__":
    main()
