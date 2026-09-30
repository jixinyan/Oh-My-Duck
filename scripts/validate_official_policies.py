import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


REVISION = "1b56c396825c052a4e26e95cf2b8d8298af9e9b4"
EXPECTED_FILES = {
    "alpha_ground_pick.onnx": "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
    "alpha_sitstand.onnx": "c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    "alpha_walking.onnx": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    "ball_kick_left.onnx": "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
    "roller.onnx": "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd",
    "roller_crouch.onnx": "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
    "roulade.onnx": "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b",
    "velstand.onnx": "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45",
}
MANIFEST_SHA256 = "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94"


def validate(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Official manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (manifest["schema_version"], manifest["model_api"], manifest["obs_len"], manifest["action_len"]) != (2, 1, 61, 14):
        raise ValueError("Official manifest API or tensor dimensions changed")
    if manifest["robot"] != {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50}:
        raise ValueError("Official robot metadata changed")
    files = {entry["file"] for entry in manifest["policies"]}
    if files != set(EXPECTED_FILES) or len(manifest["policies"]) != len(EXPECTED_FILES):
        raise ValueError("Official policy file set changed")
    results = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    for name, expected_sha in EXPECTED_FILES.items():
        path = directory / name
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"{name}: SHA-256 mismatch")
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(f"{name}: expected one input and one output")
        if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
            raise ValueError(f"{name}: tensors must be float32")
        if inputs[0].shape not in ([1, 61], ["batch", 61]) or outputs[0].shape not in ([1, 14], ["batch", 14]):
            raise ValueError(f"{name}: observation/action shape differs")
        observation = np.zeros((1, 61), dtype=np.float32)
        action = session.run([outputs[0].name], {inputs[0].name: observation})[0]
        if action.shape != (1, 14) or not np.isfinite(action).all():
            raise ValueError(f"{name}: invalid ORT output")
        results[name] = {
            "sha256": actual_sha,
            "input": {"name": inputs[0].name, "shape": inputs[0].shape, "type": inputs[0].type},
            "output": {"name": outputs[0].name, "shape": outputs[0].shape, "type": outputs[0].type},
            "metadata": session.get_modelmeta().custom_metadata_map,
            "zero_action": action[0].tolist(),
        }
    return {"repo": "pollen-robotics/microduck-policies", "revision": REVISION,
            "manifest_sha256": MANIFEST_SHA256, "manifest": manifest, "models": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
