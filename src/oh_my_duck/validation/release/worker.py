import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import urlsplit

from oh_my_duck.core.paths import project_root
from oh_my_duck.experience.harness_replay import save_json
from oh_my_duck.validation.release.plans import validate_plan


def main():
    parser = argparse.ArgumentParser(description="检查实际远程环境、场景、policy 和 Harness 来源，不启动 CUDA")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--edh-source", type=Path, required=True)
    parser.add_argument("--harness-manifest", type=Path, required=True)
    parser.add_argument("--provider-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    args.plan = args.plan if args.plan.is_absolute() else root / args.plan
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "scope": "Actual runtime metadata, source pins, policy and asset checks",
              "cuda_runtime_checked": False, "scenes": []}
    try:
        plan, scenes, inputs = validate_plan(root, args.plan)
        result.update(stages=len(plan["stages"]), input_sha256=inputs,
            source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
        manifest = json.loads(args.harness_manifest.read_text())
        revision = manifest["revision"]
        if revision != "8a5e685b22d032207f53db20454f0992a4ad60fd" or not manifest["files"]:
            raise ValueError("Native Harness manifest differs from the pinned source")
        for name, expected in manifest["files"].items():
            source = Path(name)
            if source.is_absolute() or ".." in source.parts:
                raise ValueError("Native Harness source file must belong to its dependency directory")
            if hashlib.sha256((args.edh_source / source).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Native Harness source differs: {source}")
        (args.edh_source / "harness/contracts/schema/physical.schema.json").resolve(strict=True)
        result["harness_revision"] = revision
        result["harness_source_files_verified"] = len(manifest["files"])
        provider_data = tomllib.loads(args.provider_config.read_text())
        provider = provider_data["model_providers"][provider_data["model_provider"]]
        endpoint = urlsplit(provider["base_url"])
        if (endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or
                endpoint.password or endpoint.query or endpoint.fragment):
            raise ValueError("Model provider requires a valid endpoint")
        result["model"] = provider_data["model"]
        result["model_api"] = provider.get("wire_api", "chat")
        for index, name in enumerate(scenes):
            output = args.output / f"scene-{index}.json"
            subprocess.run([sys.executable, str(root / "omd.py"), "doctor", "--runtime", "isaac-newton",
                "--metadata-only", "--scene-config", str(root / name), "--catalog", str(args.catalog),
                "--output", str(output)], cwd=root, check=True, stdout=subprocess.DEVNULL)
            checks = json.loads(output.read_text())
            if not checks["passed"] or checks["cuda_runtime_checked"]:
                raise AssertionError("Metadata preflight must pass without CUDA initialization")
            result["scenes"].append({"configuration": name, "checks": checks,
                                     "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        result["passed"] = True
    finally:
        failure = sys.exception()
        if failure is not None:
            result["error_type"] = type(failure).__name__
            result["error"] = str(failure)
        save_json(args.output / "result.json", result)
    print(json.dumps({"passed": result["passed"], "stages": result["stages"], "scenes": len(result["scenes"]),
                      "cuda_runtime_checked": False}))


if __name__ == "__main__":
    main()
