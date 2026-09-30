import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from oh_my_duck.robotics.microduck.official_policies import (
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    OfficialPolicyCatalogue,
    POLICY_SHA256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.directory or root / ".cache" / "official-policies" / OFFICIAL_REVISION
    if args.local_files_only:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    else:
        snapshot_download(repo_id=OFFICIAL_REPO, revision=OFFICIAL_REVISION,
                          allow_patterns=["manifest.json", *POLICY_SHA256],
                          local_dir=directory)
    catalogue = OfficialPolicyCatalogue(directory)
    print(json.dumps({"repo": OFFICIAL_REPO, "revision": OFFICIAL_REVISION,
                      "directory": str(directory.resolve()),
                      "policy_count": len(catalogue.policies),
                      "policies": catalogue.describe_apartment()}, indent=2))


if __name__ == "__main__":
    main()
