import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from oh_my_duck.rl.backends.isaac_newton.paths import asset_dir, require_asset, ROBOT_MODELS


def main():
    parser = argparse.ArgumentParser(description="Build or reuse source-verified Microduck assets")
    parser.add_argument("--model", choices=list(ROBOT_MODELS), default="walk")
    parser.add_argument("--reuse-from-source", type=Path,
                        help="Reuse verified assets from a clean source checkout with identical conversion inputs")
    parser.add_argument("--accept-eula", action="store_true",
        help="Accept the NVIDIA Omniverse EULA for this conversion process (OMNI_KIT_ACCEPT_EULA=YES)")
    args = parser.parse_args()
    if args.reuse_from_source is not None:
        from oh_my_duck.rl.backends.isaac_newton.asset_reuse import reuse_asset

        print(json.dumps(reuse_asset(args.reuse_from_source, args.model), allow_nan=False))
        return 0
    if (asset_dir(args.model) / "build.json").is_file():
        print("Verified cached asset:", require_asset(args.model))
        return 0
    child_env = os.environ.copy()
    if args.accept_eula:
        child_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    # 使用固定 Kit 的启动配置检查 EULA 确认状态。
    marker = Path(sys.prefix) / "lib/python3.12/site-packages/isaacsim/kit/EULA_ACCEPTED"
    accepted = child_env.get("OMNI_KIT_ACCEPT_EULA", "").lower() in {"y", "yes", "1"}
    prior = marker.is_file() and marker.read_text().split("\n", 1)[0].strip().lower() in {"y", "yes", "1"}
    if not accepted and not prior:
        parser.error("Asset conversion requires NVIDIA Omniverse EULA acceptance. Review "
            "https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html "
            "and pass --accept-eula or set OMNI_KIT_ACCEPT_EULA=YES. Headless jobs cannot answer Kit's prompt.")
    result = subprocess.run([sys.executable, "-m", "oh_my_duck.rl.backends.isaac_newton.convert_asset", "--model", args.model], env=child_env)
    if result.returncode:
        return result.returncode
    print("Verified converted asset:", require_asset(args.model))
    return 0

if __name__ == "__main__":
    sys.exit(main())
