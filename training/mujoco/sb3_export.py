"""Export native SB3 artifacts through official run_export and runner ONNX export.

The registered runner extension owns the SB3-specific actor/normalizer graph.
Official task construction, metadata, and OnPolicyRunner export remain unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from rsl_rl.runners import OnPolicyRunner
import mjlab.tasks
from mjlab.tasks.registry import register_mjlab_task, load_env_cfg, load_rl_cfg, list_tasks
from omd_microduck.export import ExportConfig, run_export
from omd_microduck.publish.manifest import check_onnx, smoke_run_onnx


class NormalizedSB3Actor(torch.nn.Module):
    def __init__(self, policy, normalizer):
        super().__init__()
        self.policy = deepcopy(policy).cpu().eval()
        if not normalizer.norm_obs or normalizer.norm_reward:
            raise ValueError("Expected observation normalization ON, reward normalization OFF")
        self.register_buffer("mean", torch.as_tensor(normalizer.obs_rms.mean, dtype=torch.float64))
        self.register_buffer("variance", torch.as_tensor(normalizer.obs_rms.var, dtype=torch.float64))
        self.epsilon = float(normalizer.epsilon)
        self.clip = float(normalizer.clip_obs)

    def forward(self, obs):
        # VecNormalize evaluates with float64 running statistics then casts to
        # float32. Preserve that order, including the clip, in the ONNX graph.
        normalized = ((obs.to(torch.float64) - self.mean) / torch.sqrt(self.variance + self.epsilon))
        normalized = torch.clamp(normalized, -self.clip, self.clip).to(torch.float32)
        return self.policy._predict(normalized, deterministic=True)

    def as_onnx(self, verbose=False):
        return self

    def get_dummy_inputs(self):
        return (torch.zeros(1, 61),)

    input_names = ["obs"]
    output_names = ["actions"]


class SB3ExportRunner(OnPolicyRunner):
    """Inference-only runner binding; inherits the official ONNX export method."""
    def __init__(self, env, train_cfg, device="cpu", **kwargs):
        self.env = env
        self.alg = None

    def load(self, path, map_location=None):
        source = Path(path)
        run = json.loads((source.parent / "run.json").read_text())
        for name in (source.name, "vecnormalize.pkl"):
            actual = hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
            if actual != run["files"][name]:
                raise ValueError(f"Artifact hash differs from saved run: {name}")
        model = PPO.load(source, device="cpu")
        # Native VecNormalize pickle is produced by our own training entry point.
        with (source.parent / "vecnormalize.pkl").open("rb") as stream:
            normalizer = pickle.load(stream)
        actor = NormalizedSB3Actor(model.policy, normalizer)
        self.alg = SimpleNamespace(get_policy=lambda: actor)

    def get_inference_policy(self, device=None):
        return self.alg.get_policy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = json.loads((args.run / "run.json").read_text())
    task = run["task"]
    if run["backend"] != "mujoco" or run["framework"] != "sb3":
        parser.error("Expected a MuJoCo SB3 run")
    if task not in list_tasks() or "MicroDuck" not in task:
        parser.error("Source run must name a pinned official Microduck task")
    alias = "Omd-SB3-Export-" + task
    register_mjlab_task(alias, load_env_cfg(task), load_env_cfg(task, play=True), load_rl_cfg(task), SB3ExportRunner)
    args.output.mkdir(parents=True, exist_ok=False)
    result = run_export(alias, ExportConfig(checkpoint_file=str(args.run / "model.zip"),
        onnx_file=str(args.output / "policy.onnx"), num_envs=1, device=args.device))
    check_onnx(result.onnx_path)
    smoke_run_onnx(result.onnx_path)
    model = PPO.load(args.run / "model.zip", device="cpu")
    with (args.run / "vecnormalize.pkl").open("rb") as stream:
        normalizer = pickle.load(stream)
    batch = np.random.default_rng(42).normal(size=(32, 61)).astype(np.float32)
    batch[:, 3:6] = (0, 0, -1)
    # Include outliers to exercise the normalization clip boundary explicitly.
    batch[16:] *= 1000
    expected, _ = model.predict(normalizer.normalize_obs(batch.copy()), deterministic=True)
    session = ort.InferenceSession(str(result.onnx_path), providers=["CPUExecutionProvider"])
    actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in batch])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-5)
    report = {"status": "passed", "task": task, "framework": "sb3", "backend": "mujoco",
        "export_path": "omd_microduck.export.run_export -> inherited OnPolicyRunner.export_policy_to_onnx",
        "normalizer": "native SB3 VecNormalize mean/variance/epsilon/clip baked into graph",
        "samples": len(batch), "max_abs_error": float(np.max(np.abs(actual-expected))),
        "policy_sha256": hashlib.sha256(result.onnx_path.read_bytes()).hexdigest(),
        "source_run": str(args.run.resolve()), "behavior_validation": "pending"}
    (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
