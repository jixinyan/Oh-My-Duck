"""Build assets in a child process and verify artifacts independently of Kit teardown."""
import argparse
import subprocess
import sys
from .paths import asset_dir, require_asset


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if (asset_dir() / "build.json").is_file():
        print("Verified cached asset:", require_asset())
        return 0
    result = subprocess.run([sys.executable, "-m", "omd_isaac.convert_asset"])
    if result.returncode:
        return result.returncode
    print("Verified converted asset:", require_asset())
    return 0

if __name__ == "__main__":
    sys.exit(main())
