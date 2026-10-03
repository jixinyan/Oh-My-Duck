import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from urllib.request import urlopen

from oh_my_duck.core.paths import project_root


def service(url, route):
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("服务检查需要本机回环地址")
    with urlopen(url.rstrip("/") + route, timeout=15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description="检查实际依赖、服务、策略和场景资产")
    parser.add_argument("--runtime", choices=("voice-client", "isaac-newton"), required=True)
    parser.add_argument("--asr-url")
    parser.add_argument("--tts-url")
    parser.add_argument("--harness-url")
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = project_root() if args.runtime == "isaac-newton" or args.scene_config is not None else None
    result = {"scope": "Installed runtime and asset/service readiness; behavioral acceptance separate",
        "runtime": args.runtime, "checks": {}}
    checks = result["checks"]
    dependencies = ({"httpx": "0.28.1", "numpy": "2.5.3", "sounddevice": "0.5.6", "soundfile": "0.14.0"}
        if args.runtime == "voice-client" else {"torch": "2.10.0+cu130", "newton": "1.2.1",
            "warp-lang": "1.13.0", "mujoco": "3.8.0", "mujoco-warp": "3.8.0.3", "jsonschema": "4.25.1"})
    for package, expected in dependencies.items():
        installed = version(package)
        if installed != expected:
            raise RuntimeError(f"{package}: 需要 {expected}，当前版本为 {installed}")
        checks[package] = installed
    if args.runtime == "isaac-newton":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        checks["cuda"] = {"version": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        expected = json.loads((root / "configs/upstream.json").read_text())["isaac_candidate"]["commit"]
        actual = subprocess.run(["git", "-C", str(root / ".cache/upstream/IsaacLab"), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        if actual != expected:
            raise RuntimeError("IsaacLab revision 与项目配置不一致")
        checks["isaaclab_revision"] = actual
    for name, url, expected in (("asr", args.asr_url, "Qwen/Qwen3-ASR-0.6B@5eb144179a02acc5e5ba31e748d22b0cf3e303b0"),
            ("tts", args.tts_url, "Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc")):
        if url is not None:
            health = service(url, "/health")
            if health != {"service": name, "model_revision": expected}:
                raise RuntimeError(f"{name} 服务的模型身份不一致")
            checks[name] = health
    if args.harness_url is not None:
        configuration = service(args.harness_url, "/api/config")
        checks["harness"] = {"reachable": True, "configuration_digest": configuration["digest"]}
    if args.scene_config is not None:
        from oh_my_duck.rl.backends.isaac_newton.paths import require_asset

        config = json.loads(args.scene_config.resolve(strict=True).read_text())
        if config["backend"] != "isaac-newton":
            raise ValueError("scene-config 必须使用 Isaac/Newton")
        checks["scene"] = {name: str((root / config[name]).resolve(strict=True))
            for name in ("usd_path", "provenance_path", "public_map_path")}
        checks["robot_asset"] = str(require_asset(config.get("robot_model", "allcollisions")))
    if args.catalog is not None:
        from oh_my_duck.robotics.microduck.official_policies import OfficialPolicyCatalogue

        catalogue = OfficialPolicyCatalogue(args.catalog)
        checks["policies"] = {name: policy.sha256 for name, policy in catalogue.policies.items()}
    result["passed"] = True
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    print(serialized, end="")
    return 0
