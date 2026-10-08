import argparse
import hashlib
from importlib.metadata import distribution
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import oh_my_duck


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-record", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--metric-campaign", type=Path)
    parser.add_argument("--metric-plan", type=Path)
    args = parser.parse_args()
    if (args.policy_record is None) != (args.catalog is None):
        raise ValueError("An actual policy record and catalogue must be provided together")
    if (args.metric_campaign is None) != (args.metric_plan is None):
        raise ValueError("An actual metric campaign and its declared plan must be provided together")
    root = args.source_root.resolve(strict=True)
    packages = args.packages.resolve(strict=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    installed = Path(oh_my_duck.__file__).resolve(strict=True)
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError("Package validation requires an independent installed environment")
    wheel, source = packages / "oh_my_duck-0.1.0-py3-none-any.whl", packages / "oh_my_duck-0.1.0.tar.gz"
    tracked = subprocess.check_output(["git", "ls-files", "src/oh_my_duck"], cwd=root, text=True).splitlines()
    required = [name for name in tracked if Path(name).suffix == ".py"
                or "/robotics/microduck/microduck/" in name
                or "/robotics/microduck/xl330_test_bench/" in name
                or "/rl/backends/isaac_newton/resources/" in name
                or name.endswith("odom_anchor_sets.json")]
    licenses = ["THIRD_PARTY_NOTICES.md", "third_party/microduck_rl/LICENSE",
                "docs/third_party/isaaclab-microduck-LICENSE.txt"]
    with zipfile.ZipFile(wheel) as archive, tarfile.open(source, "r:gz") as source_archive:
        if archive.testzip() is not None or any("__pycache__" in name or name.endswith(".pyc") for name in archive.namelist()):
            raise AssertionError("Wheel contains invalid or generated files")
        for name in required:
            expected = (root / name).read_bytes()
            if archive.read(name.removeprefix("src/")) != expected:
                raise AssertionError(f"Wheel differs from tracked source: {name}")
            member = source_archive.extractfile("oh_my_duck-0.1.0/" + name)
            if member is None or member.read() != expected:
                raise AssertionError(f"Source distribution differs from tracked source: {name}")
            installed_file = installed.parent / name.removeprefix("src/oh_my_duck/")
            if installed_file.read_bytes() != expected:
                raise AssertionError(f"Installed package differs from tracked source: {name}")
        for name in licenses:
            if archive.read("oh_my_duck-0.1.0.dist-info/licenses/" + name) != (root / name).read_bytes():
                raise AssertionError(f"Wheel license differs from source: {name}")
    metadata = distribution("oh-my-duck")
    if metadata.version != "0.1.0" or not files("oh_my_duck.robotics.microduck").joinpath("microduck/scene_apartment.xml").is_file():
        raise AssertionError("Installed metadata or Microduck scene resource is missing")
    environment = {**os.environ, "OMD_PROJECT_ROOT": str(root), "TMPDIR": str(root / ".cache/tmp"),
                   "CUDA_VISIBLE_DEVICES": ""}
    environment.pop("PYTHONPATH", None)
    cli = Path(sys.prefix) / "bin/omd"
    commands = [["--help"], ["tasks"], ["frameworks"], ["status"], ["doctor", "--runtime", "voice-client"],
                ["voice", "--help"], ["voice-task", "--help"], ["voice-session", "--help"],
                ["harness", "--help"], ["sim", "--help"], ["validate", "--help"], ["replay", "--help"],
                ["validate", "metric-audit", "--help"], ["validate", "policy-audit", "--help"],
                ["validate", "release", "--help"], ["validate", "navigation-audit", "--help"],
                ["validate", "release-plan"]]
    if args.policy_record is not None:
        commands.append(["validate", "policy-audit", "--directory", str(args.policy_record.resolve(strict=True)),
                         "--catalog", str(args.catalog.resolve(strict=True)), "--output",
                         str(args.output.resolve().with_name(args.output.stem + "-policy.json"))])
    if args.metric_campaign is not None:
        commands.append(["validate", "metric-audit", "--campaign", str(args.metric_campaign.resolve(strict=True)),
                         "--plan", str(args.metric_plan.resolve(strict=True)), "--source-root", str(root),
                         "--output", str(args.output.resolve().with_name(args.output.stem + "-metric.json"))])
    checks = []
    for arguments in commands:
        completed = subprocess.run([str(cli), *arguments], cwd=Path.home(), env=environment,
                                   check=True, capture_output=True, text=True, timeout=30)
        if not completed.stdout.strip():
            raise AssertionError(f"Installed CLI returned no result: {arguments}")
        if arguments[0] == "doctor" and json.loads(completed.stdout)["cuda_runtime_checked"]:
            raise AssertionError("Installed voice client initialized CUDA")
        checks.append({"arguments": arguments, "passed": True,
                       "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()})
    result = {"passed": True, "scope": "Built distributions and independent installed CLI outside checkout",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "installed_package": str(installed), "tracked_files_verified": len(required),
              "licenses_verified": len(licenses), "cli_checks": checks,
              "cuda_runtime_checked": False,
              "package_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (wheel, source)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": True, "tracked_files_verified": len(required), "cli_checks": len(checks)}))


if __name__ == "__main__":
    main()
