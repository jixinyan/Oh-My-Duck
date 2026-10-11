import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

from oh_my_duck.core.paths import project_root
from oh_my_duck.infrastructure.owned_process import owned_process
from oh_my_duck.integrations.native_runtime import EDH_REVISION, verify_edh_source
from oh_my_duck.robotics.microduck.policy_sources import (
    MANIFEST_SHA256, OFFICIAL_REPO, OFFICIAL_REVISION, POLICY_SHA256,
)


def run_stage(name: str, arguments: list[str], *, output: Path, cwd: Path, environment: dict) -> str:
    with (output / f"{name}.stdout.log").open("x") as stdout, (output / f"{name}.stderr.log").open("x") as stderr:
        with owned_process(arguments, record_path=output / f"{name}.process.json",
                           cwd=cwd, env=environment, stdout=stdout, stderr=stderr) as process:
            code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, arguments)
    return (output / f"{name}.stdout.log").read_text(encoding="utf-8")


def prepare_policies(directory: Path, output: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    downloads = output / "downloads"
    downloads.mkdir()
    records = {}
    for name, expected in {"manifest.json": MANIFEST_SHA256, **POLICY_SHA256}.items():
        target = directory / name
        downloaded = not target.exists()
        if downloaded:
            candidate = downloads / name
            url = f"https://huggingface.co/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/{name}"
            with urlopen(url, timeout=120) as response, candidate.open("xb") as file:
                shutil.copyfileobj(response, file)
            with candidate.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError(f"Official policy SHA256 differs: {name}")
            with candidate.open("rb") as source, target.open("xb") as file:
                shutil.copyfileobj(source, file)
        with target.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"Official policy SHA256 differs: {name}")
        records[name] = {"sha256": digest, "bytes": target.stat().st_size, "downloaded": downloaded}
    return {"repository": OFFICIAL_REPO, "revision": OFFICIAL_REVISION,
            "directory": str(directory), "files": records}


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="准备固定原生 Harness、官方策略和锁定 CPU 环境")
    parser.add_argument("--edh-repository", type=Path, help="包含固定提交的已有 EDH Git 仓库")
    parser.add_argument("--edh-source", type=Path, default=root / ".cache/edh" / EDH_REVISION)
    parser.add_argument("--policy-dir", type=Path, default=root / ".cache/official-policies" / OFFICIAL_REVISION)
    parser.add_argument("--cpu-environment", type=Path, default=root / ".cache/cpu-apartment-locked-venv")
    parser.add_argument("--skip-cpu-environment", action="store_true", help="仿真 worker 已在其他主机准备时，仅准备 Node 与策略")
    parser.add_argument("--output", type=Path, required=True, help="新的准备记录目录")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "scope": "固定源码、锁定依赖、SDK 导入与官方策略准备",
              "cuda_runtime_checked": False, "model_request_performed": False,
              "physical_task_performed": False, "cpu_environment_prepared": False}
    try:
        temp_dir = root / ".cache/tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TMPDIR": str(temp_dir),
                       "npm_config_cache": str(root / ".cache/npm"),
                       "UV_CACHE_DIR": str(root / ".cache/uv"),
                       "UV_PYTHON_INSTALL_DIR": str(root / ".cache/python"),
                       "UV_HTTP_TIMEOUT": "600", "OMD_PROJECT_ROOT": str(root)}
        environment.pop("PYTHONPATH", None)
        commands = {name: shutil.which(name) for name in ("node", "npm", "git")}
        for name, executable in commands.items():
            if executable is None:
                raise FileNotFoundError(f"Native Harness setup requires {name}")
        result["source_revision"] = subprocess.check_output(
            [commands["git"], "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        result["source_dirty"] = bool(subprocess.check_output(
            [commands["git"], "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip())
        inputs = ["pyproject.toml", "environments/cpu-apartment/pyproject.toml",
                  "environments/cpu-apartment/uv.lock", "integrations/edh/runtime-check.mjs",
                  "src/oh_my_duck/infrastructure/bootstrap_harness.py",
                  "src/oh_my_duck/integrations/native_runtime.py",
                  "src/oh_my_duck/robotics/microduck/policy_sources.py", "src/oh_my_duck/cli/doctor.py"]
        result["input_sha256"] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in inputs}
        runtime_check = root / "integrations/edh/runtime-check.mjs"
        result["node"] = json.loads(run_stage("node-version", [commands["node"], str(runtime_check),
            "--node-only"], output=output, cwd=root, environment=environment))["node"]
        source = args.edh_source.resolve()
        if not source.exists():
            if args.edh_repository is None:
                raise ValueError("Creating the pinned SDK requires --edh-repository")
            repository = args.edh_repository.resolve(strict=True)
            subprocess.run([commands["git"], "-C", str(repository), "cat-file", "-e",
                            f"{EDH_REVISION}^{{commit}}"], check=True, capture_output=True)
            source.parent.mkdir(parents=True, exist_ok=True)
            run_stage("source-clone", [commands["git"], "clone", "--no-local", "--no-checkout",
                str(repository), str(source)], output=output, cwd=root, environment=environment)
            run_stage("source-select", [commands["git"], "-C", str(source), "switch", "--detach",
                EDH_REVISION], output=output, cwd=root, environment=environment)
        source = verify_edh_source(source)
        result.update(harness_revision=EDH_REVISION, harness_source=str(source))
        package = json.loads((source / "package.json").read_text(encoding="utf-8"))
        package_manager = package["packageManager"]
        if package_manager != "pnpm@11.19.0":
            raise ValueError("Pinned Harness package manager differs")
        pnpm = [commands["npm"], "exec", "--yes", f"--package={package_manager}", "--", "pnpm"]
        result["pnpm"] = run_stage("pnpm-version", [*pnpm, "--version"],
            output=output, cwd=root, environment=environment).strip()
        if result["pnpm"] != "11.19.0":
            raise RuntimeError("Executed package manager differs from the pinned SDK version")
        run_stage("node-dependencies", [*pnpm, "install", "--frozen-lockfile"],
            output=output, cwd=source, environment=environment)
        verify_edh_source(source)
        node_environment = {**environment, "TSX_TSCONFIG_PATH": str(source / "tsconfig.runtime.json")}
        loader = (source / "node_modules/tsx/dist/loader.mjs").resolve(strict=True)
        result["native_sdk"] = json.loads(run_stage("native-sdk", [commands["node"], "--import",
            str(loader), str(runtime_check), str(source)], output=output, cwd=root, environment=node_environment))
        result["policies"] = prepare_policies(args.policy_dir.resolve(), output)
        if not args.skip_cpu_environment:
            uv = shutil.which("uv")
            if uv is None:
                raise FileNotFoundError("CPU environment setup requires uv")
            cpu_environment = args.cpu_environment.resolve()
            cpu_variables = {**environment, "UV_PROJECT_ENVIRONMENT": str(cpu_environment)}
            run_stage("cpu-dependencies", [uv, "sync", "--project", str(root / "environments/cpu-apartment"),
                "--locked", "--no-dev", "--no-editable", "--python", "3.12"],
                output=output, cwd=root, environment=cpu_variables)
            python = (cpu_environment / "bin/python").absolute()
            run_stage("python-sdk", [uv, "pip", "install", "--python", str(python), "--no-deps",
                "--reinstall-package", "edh-physical-harness", str(source / "harness/physical-runtime")],
                output=output, cwd=root, environment=environment)
            verify_edh_source(source)
            result["cpu_runtime"] = json.loads(run_stage("cpu-runtime", [str(python),
                "-m", "oh_my_duck", "doctor", "--runtime", "cpu-apartment", "--catalog",
                str(args.policy_dir.resolve()), "--edh-source", str(source)],
                output=output, cwd=output, environment=environment))
            result.update(cpu_environment_prepared=True, cpu_environment=str(cpu_environment))
        result["passed"] = True
    finally:
        failure = sys.exception()
        if failure is not None:
            result.update(error_type=type(failure).__name__, error=str(failure))
        with (output / "result.json").open("x", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
