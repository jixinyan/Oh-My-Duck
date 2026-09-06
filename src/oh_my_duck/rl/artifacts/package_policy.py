"""Create a local official schema-2 package; this entry point has no upload mode."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from oh_my_duck.rl.artifacts.publish.cli import PublishConfig, run
from oh_my_duck.rl.artifacts.publish import manifest as official
from oh_my_duck.core.paths import project_root
from oh_my_duck.rl.training.tasks import project_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", help="Optional consistency check against saved training task")
    args = parser.parse_args()
    source, checkpoint, output = args.onnx.resolve(), args.checkpoint.resolve(), args.output.resolve()
    if output.exists():
        parser.error("Output already exists; retain previous evidence")
    from .identity import verify_export_source

    verify_export_source(source, checkpoint)
    run_manifest = json.loads((checkpoint.parent / "run.json").read_text())
    task = project_tasks().get(run_manifest["task"])
    if args.task is not None and args.task != task.id:
        parser.error("Task differs from checkpoint run")
    if task.policy_package is None:
        parser.error("Task has no explicit deployment profile")
    if run_manifest["backend"] not in task.bindings:
        parser.error("Run backend is not registered for this task")
    if run_manifest["framework"] not in ("rsl-rl", "sb3"):
        parser.error("Unsupported checkpoint framework")
    profile = task.policy_package.build()
    original = Path.cwd()
    repo = "jixinyan/microduck-" + profile.name
    with tempfile.TemporaryDirectory(prefix="omd-policy-") as temporary:
        try:
            os.chdir(temporary)
            run(
                PublishConfig(
                    repo=repo,
                    name=profile.name,
                    onnx=str(source),
                    kind=profile.kind,
                    slot=profile.slot,
                    duration_s=profile.duration_s,
                    entry_pose=profile.entry_pose,
                    dry_run=True,
                    description="Local training artifact; behavior and hardware acceptance are separate",
                )
            )
        finally:
            os.chdir(original)
        staged = Path(temporary) / ("publish-" + profile.name)
        manifest = json.loads((staged / "manifest.json").read_text())
        manifest["training"].update(
            task_id=task.id,
            checkpoint_file=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            framework=run_manifest["framework"],
            backend=run_manifest["backend"],
            extension_repo="jixinyan/Oh-My-Duck",
            extension=official.git_provenance(project_root()),
        )
        manifest["validation"] = {
            "format": "official_shape_smoke_and_schema_passed",
            "behavior": "unvalidated",
            "hardware": "unvalidated",
            "policy_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        official.validate_manifest(manifest)
        (staged / "manifest.json").write_text(official.dump_manifest(manifest))
        (staged / "README.md").write_text(
            "# Local "
            + profile.name
            + " policy\n\nFormat validation does not establish learned behavior or hardware compatibility. No upload has been made.\n\n"
            + official.render_readme(manifest, repo)
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged, output)
    print("Local official-format package:", output)


if __name__ == "__main__":
    main()
