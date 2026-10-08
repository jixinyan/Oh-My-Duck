from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tomllib
from urllib.parse import urlsplit

from oh_my_duck.core.paths import project_root


EDH_REVISION = "8a5e685b22d032207f53db20454f0992a4ad60fd"
POLICY_REVISION = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"


def main() -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run a native EDH MicroDuck simulation deployment")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--provider-config", type=Path)
    source.add_argument("--remote-provider-config", type=str)
    parser.add_argument("--ssh-host", type=str)
    parser.add_argument("--remote-python", type=str)
    parser.add_argument("--edh-source", type=Path,
                        default=root / ".cache/edh" / EDH_REVISION)
    parser.add_argument("--cpu-python", type=Path,
                        default=root / ".cache/cpu-apartment-locked-venv/bin/python")
    parser.add_argument("--simulation-python", type=Path)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model-api", choices=("chat-completions", "responses"))
    parser.add_argument("--check-model", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max"))
    parser.add_argument("--worker-host", type=str)
    parser.add_argument("--worker-root", type=str)
    parser.add_argument("--worker-python", type=str)
    parser.add_argument("--worker-edh-source", type=str)
    parser.add_argument("--worker-policy-dir", type=str)
    parser.add_argument("--worker-policy-registry", type=str)
    parser.add_argument("--worker-cuda-device", type=int)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--policy-dir", type=Path,
                        default=root / ".cache/official-policies" / POLICY_REVISION)
    parser.add_argument("--policy-registry", type=Path,
                        help="Registered schema-2 policy packages on the local simulation host")
    parser.add_argument("--data-dir", type=Path,
                        default=root / ".cache/harness-data")
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.policy_registry is not None and args.worker_policy_registry is not None:
        raise ValueError("Choose a local or remote policy registry")
    if args.policy_registry is not None and args.worker_host is not None:
        raise ValueError("Remote simulation requires --worker-policy-registry")
    if args.worker_policy_registry is not None:
        registry_path = PurePosixPath(args.worker_policy_registry)
        if args.worker_host is None or not registry_path.is_absolute() or ".." in registry_path.parts:
            raise ValueError("Remote registry requires --worker-host and an absolute path without parent components")
    edh_source = args.edh_source.resolve(strict=True)
    revision = subprocess.run(
        ["git", "-C", str(edh_source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != EDH_REVISION:
        raise ValueError("EDH source revision differs from the pinned native runtime")
    changed = subprocess.run(
        ["git", "-C", str(edh_source), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    ).stdout
    if changed:
        raise RuntimeError("Pinned EDH source contains tracked changes")
    if args.remote_provider_config is not None:
        if not args.ssh_host or not args.remote_python:
            raise ValueError("Remote provider configuration requires an SSH host and Python 3.11+")
        code = (
            "import json,os,tomllib,sys; "
            "d=tomllib.loads(open(sys.argv[1],'rb').read().decode()); "
            "p=d['model_providers'][d['model_provider']]; "
            "assert not ('experimental_bearer_token' in p and 'env_key' in p); "
            "t=p['experimental_bearer_token'] if 'experimental_bearer_token' in p "
            "else os.environ[p['env_key']] if 'env_key' in p else None; "
            "print(json.dumps({'model':d['model'],'base_url':p['base_url'],"
            "'token':t,'wire_api':p.get('wire_api','chat')}))"
        )
        remote = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.ssh_host,
             f"{shlex.quote(args.remote_python)} -c {shlex.quote(code)} "
             f"{shlex.quote(args.remote_provider_config)}"],
            check=True, capture_output=True, text=True,
        )
        selected = json.loads(remote.stdout)
        model, base_url, token = (selected[key] for key in ("model", "base_url", "token"))
        model_api = "responses" if selected["wire_api"] == "responses" else "chat-completions"
    elif args.provider_config is None:
        model = os.environ["EDH_MODEL"]
        base_url = os.environ["EDH_MODEL_BASE_URL"]
        token = os.environ.get("EDH_MODEL_API_KEY")
        model_api = os.environ.get("EDH_MODEL_API", "chat-completions")
    else:
        provider_data = tomllib.loads(args.provider_config.resolve(strict=True).read_text())
        provider = provider_data["model_providers"][provider_data["model_provider"]]
        model = provider_data["model"]
        base_url = provider["base_url"]
        model_api = "responses" if provider.get("wire_api") == "responses" else "chat-completions"
        if "experimental_bearer_token" in provider and "env_key" in provider:
            raise ValueError("Provider has conflicting credential sources")
        token = (provider["experimental_bearer_token"]
                 if "experimental_bearer_token" in provider
                 else os.environ[provider["env_key"]] if "env_key" in provider else None)
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Provider base URL must be an HTTP(S) endpoint without URL credentials")
    if token is not None and (not isinstance(token, str) or not token or
                              any(char.isspace() for char in token)):
        raise ValueError("Provider credential is missing or invalid")
    if args.model is not None:
        if not args.model.strip() or any(char.isspace() for char in args.model):
            raise ValueError("Model ID must be nonempty and contain no whitespace")
        model = args.model
    model_api = args.model_api or model_api
    if model_api not in ("chat-completions", "responses"):
        raise ValueError("Model API must be chat-completions or responses")
    remote_worker = None
    worker_paths = (args.worker_root, args.worker_python, args.worker_edh_source,
                    args.worker_policy_dir)
    if args.worker_host is not None:
        if (not all(worker_paths) or args.worker_host.startswith("-") or
                any(char.isspace() for char in args.worker_host)):
            raise ValueError("Remote worker requires an SSH host and all four absolute paths")
        for value in worker_paths:
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("Remote worker paths must be absolute without parent components")
        if args.worker_cuda_device is not None and args.worker_cuda_device < 0:
            raise ValueError("CUDA device index must be nonnegative")
        remote_worker = {
            "host": args.worker_host, "root": args.worker_root,
            "python": args.worker_python, "edh_source": args.worker_edh_source,
            "policy_dir": args.worker_policy_dir, "cuda_device": args.worker_cuda_device,
        }
    elif any(value is not None for value in (*worker_paths, args.worker_cuda_device)):
        raise ValueError("Remote worker options require --worker-host")
    simulation_python = (args.simulation_python or args.cpu_python).absolute()
    if remote_worker is None and not simulation_python.is_file():
        raise FileNotFoundError(simulation_python)
    scene_configuration = None
    if args.scene_config is not None:
        scene_path = args.scene_config.resolve(strict=True)
        scene_configuration = json.loads(scene_path.read_text(encoding="utf-8"))
        if not isinstance(scene_configuration, dict):
            raise ValueError("Scene configuration must be a JSON object")
        if scene_configuration.get("backend") != "isaac-newton":
            raise ValueError("External scenes require the Isaac Newton backend")
        for field in ("usd_path", "provenance_path", "public_map_path"):
            if field in scene_configuration:
                value = Path(scene_configuration[field])
                if remote_worker is None:
                    scene_configuration[field] = str((root / value).resolve(strict=True))
                else:
                    path = PurePosixPath(str(value))
                    if ".." in path.parts:
                        raise ValueError("Scene paths must not contain parent components")
                    scene_configuration[field] = str(PurePosixPath(args.worker_root) / path)
        if remote_worker is not None and args.worker_cuda_device is None:
            raise ValueError("Remote Isaac Newton scenes require an explicit CUDA device")
    catalog = (args.policy_dir.resolve(strict=True) if remote_worker is None
               else PurePosixPath(args.worker_policy_dir))
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = root / ".cache/tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    loader = (edh_source / "node_modules/tsx/dist/loader.mjs").resolve(strict=True)
    node_command = shutil.which("node")
    if node_command is None:
        raise FileNotFoundError("Node.js is required for the pinned EDH runtime")
    node = Path(node_command).resolve(strict=True)
    environment = dict(os.environ)
    environment.update({
        "OMD_EDH_SOURCE": str(edh_source),
        "OMD_CPU_PYTHON": str(simulation_python),
        "OMD_POLICY_DIR": str(catalog),
        "OMD_DATA_DIRECTORY": str(data_dir),
        "OMD_EDH_PORT": str(args.port),
        "OMD_SCENE_SEED": str(args.seed),
        "EDH_MODEL_BASE_URL": base_url,
        "EDH_MODEL": model,
        "EDH_MODEL_API": model_api,
        "OMD_CHECK_MODEL": "1" if args.check_model else "0",
        "TSX_TSCONFIG_PATH": str(edh_source / "tsconfig.runtime.json"),
        "TMPDIR": str(temp_dir),
    })
    registry = (str(args.policy_registry.resolve(strict=True)) if args.policy_registry is not None
                else args.worker_policy_registry)
    if registry is None:
        environment.pop("OMD_POLICY_REGISTRY", None)
    else:
        environment["OMD_POLICY_REGISTRY"] = registry
    if remote_worker is None:
        environment.pop("OMD_REMOTE_WORKER", None)
    else:
        environment["OMD_REMOTE_WORKER"] = json.dumps(remote_worker, allow_nan=False)
    if args.reasoning_effort is None:
        environment.pop("EDH_REASONING_EFFORT", None)
    else:
        environment["EDH_REASONING_EFFORT"] = args.reasoning_effort
    if scene_configuration is None:
        environment.pop("OMD_SCENE_CONFIGURATION", None)
    else:
        environment["OMD_SCENE_CONFIGURATION"] = json.dumps(
            scene_configuration, allow_nan=False)
    if token is None:
        environment.pop("EDH_MODEL_API_KEY", None)
    else:
        environment["EDH_MODEL_API_KEY"] = token
    os.execve(str(node), [str(node), "--import", str(loader),
                          str(root / "integrations/edh/server.mjs")], environment)


if __name__ == "__main__":
    raise SystemExit(main())
