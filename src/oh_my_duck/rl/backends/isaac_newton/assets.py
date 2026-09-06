"""Build assets in a child process and verify artifacts independently of Kit teardown."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from oh_my_duck.rl.backends.isaac_newton.paths import asset_dir, require_asset, ROBOT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(ROBOT_MODELS), default="walk")
    parser.add_argument("--accept-eula", action="store_true",
        help="Accept the NVIDIA Omniverse EULA for this conversion process (OMNI_KIT_ACCEPT_EULA=YES)")
    args = parser.parse_args()
    if (asset_dir(args.model) / "build.json").is_file():
        print("Verified cached asset:", require_asset(args.model))
        return 0
    child_env = os.environ.copy()
    if args.accept_eula:
        child_env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    # Match the pinned Kit bootstrap without importing Isaac Sim or prompting on a worker.
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
