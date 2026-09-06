"""Export a diagnostic checkpoint and compare Torch and ONNX on identical inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .contracts import HOME, JOINT_NAMES, OBSERVATION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import launch_simulation
    from .agent import DiagnosticRunnerCfg
    from .config import DiagnosticEnvCfg
    from .environment import DiagnosticEnv
    cfg = DiagnosticEnvCfg()
    cfg.scene.num_envs = 1
    report = {"status": "failed", "validation": "unvalidated", "actuator": "diagnostic_pd"}
    env = None
    try:
        with launch_simulation(cfg, {"headless": True}):
            env = DiagnosticEnv(cfg)
            runner = OnPolicyRunner(RslRlVecEnvWrapper(env), DiagnosticRunnerCfg().to_dict(), device=env.device)
            runner.load(str(args.checkpoint), map_location=env.device)
            model = runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
            destination = args.output / "policy.onnx"
            # Use the library's exportable actor, including its observation normalizer.
            torch.onnx.export(model, model.get_dummy_inputs(), str(destination), opset_version=18,
                input_names=model.input_names, output_names=model.output_names, dynamo=False)
            graph = onnx.load(str(destination))
            metadata = {"joint_names": ",".join(JOINT_NAMES), "default_joint_pos": ",".join(map(str, HOME)),
                "action_scale": "1.0", "observation_names": ",".join(OBSERVATION_NAMES),
                "command_names": "twist,head_pose,body_pose", "actuator": "diagnostic_pd", "validation": "unvalidated"}
            for key, value in metadata.items():
                entry = graph.metadata_props.add(); entry.key = key; entry.value = value
            onnx.save(graph, str(destination))
            onnx.checker.check_model(str(destination))
            session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
            generator = torch.Generator().manual_seed(0)
            maximum = 0.0
            for i in range(16):
                sample = torch.zeros((1, 61)) if i == 0 else torch.randn((1, 61), generator=generator) * 0.1
                with torch.inference_mode():
                    expected = model(sample).numpy()
                actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
                maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            report.update(status="passed", numerical_samples=16, max_abs_error=maximum,
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                policy_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if env is not None:
            env.close()
        (args.output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1

if __name__ == "__main__":
    sys.exit(main())
