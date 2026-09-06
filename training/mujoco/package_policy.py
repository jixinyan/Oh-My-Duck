"""Create a local schema-2 policy package through the official publisher dry-run.

Only the migrated constant-command walking task is accepted. This entry point has
no upload mode; a format-valid smoke checkpoint remains explicitly unvalidated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from omd_microduck.publish.cli import PublishConfig, run
from omd_microduck.publish import manifest as official

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["Mjlab-Velocity-Flat-MicroDuck"], default="Mjlab-Velocity-Flat-MicroDuck")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    source, output, checkpoint = args.onnx.resolve(), args.output.resolve(), args.checkpoint.resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint file is required for provenance")
    if output.exists():
        parser.error("Output already exists; use a new path to retain prior evidence")
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="omd-policy-") as temporary:
        try:
            os.chdir(temporary)
            run(PublishConfig(repo="jixinyan/microduck-walking-smoke", name="walking-smoke",
                onnx=str(source), kind="perpetual", slot="walk", dry_run=True,
                description="Integration smoke checkpoint; walking quality and hardware compatibility unvalidated"))
        finally:
            os.chdir(original)
        staged = Path(temporary) / "publish-walking-smoke"
        manifest = json.loads((staged / "manifest.json").read_text())
        manifest["training"].update(task_id=args.task, checkpoint_file=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            framework="rsl-rl", backend="mujoco", extension_repo="jixinyan/Oh-My-Duck",
            extension=official.git_provenance(ROOT))
        manifest["validation"] = {"format": "official_shape_smoke_and_schema_passed",
            "behavior": "unvalidated", "hardware": "unvalidated",
            "policy_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        official.validate_manifest(manifest)
        (staged / "manifest.json").write_text(official.dump_manifest(manifest))
        (staged / "README.md").write_text(
            "# Walking integration smoke artifact\n\n"
            "This checkpoint has only five training iterations. It is not a learned gait or a hardware-validated policy. "
            "The official publisher passed format and numerical smoke checks only. No Hub upload has been made.\n\n"
            + official.render_readme(manifest, "jixinyan/microduck-walking-smoke"))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, output)
    print(f"Local official-format package: {output}")


if __name__ == "__main__":
    main()
