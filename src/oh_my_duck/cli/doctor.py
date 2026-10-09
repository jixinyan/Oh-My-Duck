import argparse
import hashlib
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


def scene_assets(root, config):
    paths = {name: (root / config[name]).resolve(strict=True)
             for name in ("usd_path", "provenance_path", "public_map_path")}
    manifest = json.loads(paths["provenance_path"].read_text())
    scene_root = paths["provenance_path"].parent / "source"
    files = manifest["files"]
    if not files or len({entry["key"] for entry in files}) != len(files):
        raise ValueError("场景来源文件清单必须包含唯一的文件路径")
    for entry in files:
        relative = Path(entry["key"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("场景来源文件路径必须位于 source 目录")
        path = (scene_root / relative).resolve(strict=True)
        if not path.is_relative_to(scene_root.resolve(strict=True)) or path.stat().st_size != entry["size"]:
            raise ValueError("场景来源文件路径或长度不一致")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError(f"场景文件 SHA256 不一致: {relative}")
    expected_usd = (scene_root / manifest["source"]["usd_relative_path"]).resolve(strict=True)
    if expected_usd != paths["usd_path"]:
        raise ValueError("场景 USD 与来源清单不一致")
    with expected_usd.open("rb") as source:
        usd_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    public_map = json.loads(paths["public_map_path"].read_text())
    if (public_map["scene_id"] != manifest["scene_id"] or public_map["frame_id"] != "world" or
            not public_map["ground_collider_paths"]):
        raise ValueError("场景 map 身份、坐标或 ground collider 信息不完整")
    if "source_usd_sha256" in public_map:
        if public_map["source_usd_sha256"] != usd_sha256:
            raise ValueError("场景 map 与 USD 的 SHA256 不一致")
    elif Path(public_map["source_usd"]).resolve(strict=True) != expected_usd:
        raise ValueError("场景 map 与 USD 路径不一致")
    return {**{key: str(value) for key, value in paths.items()}, "files_verified": len(files),
            "source_usd_sha256": usd_sha256,
            "provenance_sha256": hashlib.sha256(paths["provenance_path"].read_bytes()).hexdigest(),
            "public_map_sha256": hashlib.sha256(paths["public_map_path"].read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description="检查实际依赖、服务、策略和场景资产")
    parser.add_argument("--runtime", choices=("voice-client", "isaac-newton"), required=True)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--asr-url")
    parser.add_argument("--tts-url")
    parser.add_argument("--harness-url")
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.metadata_only and args.runtime != "isaac-newton":
        raise ValueError("metadata-only 适用于 Isaac/Newton 环境检查")
    root = project_root() if args.runtime == "isaac-newton" or args.scene_config is not None else None
    result = {"scope": "Installed runtime and asset/service readiness; behavioral acceptance separate",
        "runtime": args.runtime, "cuda_runtime_checked": False, "checks": {}}
    checks = result["checks"]
    dependencies = ({"httpx": "0.28.1", "numpy": "2.5.3", "sounddevice": "0.5.6", "soundfile": "0.14.0"}
        if args.runtime == "voice-client" else {"torch": "2.10.0+cu130", "newton": "1.2.1",
            "warp-lang": "1.13.0", "mujoco": "3.8.0", "mujoco-warp": "3.8.0.3", "jsonschema": "4.25.1"})
    for package, expected in dependencies.items():
        installed = version(package)
        if installed != expected:
            raise RuntimeError(f"{package}: 需要 {expected}，当前版本为 {installed}")
        checks[package] = installed
    if args.runtime == "voice-client":
        import sounddevice as sd
        import soundfile as sf

        portaudio_version, portaudio_description = sd.get_portaudio_version()
        checks["audio_libraries"] = {
            "portaudio_version": portaudio_version,
            "portaudio_description": portaudio_description,
            "libsndfile_version": sf.__libsndfile_version__,
        }
    if args.runtime == "isaac-newton":
        from oh_my_duck.infrastructure.usd_runtime import verify_usd_runtime

        checks["openusd"] = verify_usd_runtime()
        if not args.metadata_only:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用")
            checks["cuda"] = {"version": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
            result["cuda_runtime_checked"] = True
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
        checks["scene"] = scene_assets(root, config)
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
