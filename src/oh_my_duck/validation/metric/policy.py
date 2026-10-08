import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from oh_my_duck.robotics.policies.catalogue import PolicyCatalogue


def verify(directory: Path, catalog: Path, registry: Path | None = None) -> dict:
    catalogue = PolicyCatalogue(catalog, registry)
    samples_path = directory / "samples.json"
    events_path = directory / "events.json"
    samples = json.loads(samples_path.read_text())
    events = json.loads(events_path.read_text())
    inputs, policies, maximum_error = {}, set(), 0.0
    for sample in samples:
        control = sample["control"]
        if control is None or control["executed_actions"] == 0:
            continue
        inference = control["policy_inference"]
        if ((inference["episode_id"], inference["result_sequence"]) != (sample["episode_id"], sample["sequence"])
                or inference["sequence"] + 1 != sample["sequence"]
                or inference["raw_sim_steps"] != control["raw_sim_steps"]
                or inference["action"] != control["action"] or inference["complete"]):
            raise AssertionError("Policy evidence differs from its admitted physical control")
        policy = catalogue.policies[inference["policy"]]
        if policy.sha256 != inference["policy_sha256"]:
            raise AssertionError("Policy evidence differs from the verified ONNX")
        observation = np.asarray(inference["observation"], dtype=np.float32)
        action = np.asarray(inference["action"], dtype=np.float32)
        command = np.asarray(inference["command_block"], dtype=np.float32)
        if observation.shape != (61,) or action.shape != (14,) or command.shape != (13,):
            raise AssertionError("Policy evidence dimensions must be 61, 14 and 13")
        if not np.isfinite(observation).all() or not np.isfinite(action).all() or not np.isfinite(command).all():
            raise AssertionError("Policy evidence contains nonfinite values")
        np.testing.assert_array_equal(observation[-13:], command)
        np.testing.assert_array_equal(control["body_twist_world_after_action"], sample["body_twist_world"])
        recomputed = catalogue.infer(policy, observation)
        error = float(np.max(np.abs(action - recomputed)))
        np.testing.assert_allclose(action, recomputed, rtol=0, atol=1e-6)
        maximum_error = max(maximum_error, error)
        key = (inference["episode_id"], inference["result_sequence"])
        if key in inputs and inputs[key] != inference:
            raise AssertionError("Repeated physical observation changed its policy evidence")
        inputs[key] = inference
        policies.add(policy.name)
    if not inputs:
        raise AssertionError("Actual admitted policy controls are required")
    serialized = 0
    for event in events:
        if event["event"] != "update":
            continue
        control = event["data"].get("control")
        if control is None or control["executed_actions"] == 0:
            continue
        inference = control["policy_inference"]
        key = (inference["episode_id"], inference["result_sequence"])
        if inference != inputs[key]:
            raise AssertionError("Serialized native update differs from its actual policy evidence")
        serialized += 1
    if serialized < len(inputs):
        raise AssertionError("Serialized updates omit admitted policy evidence")
    sequences = sorted(key[1] for key in inputs)
    if sequences != list(range(1, sequences[-1] + 1)):
        raise AssertionError("Policy evidence omits a physical control sequence")
    return {"passed": True, "scope": "Recorded actual policy inputs, ONNX parity and native control publication",
            "catalogue_revision": catalogue.revision,
            "control_steps": len(inputs), "serialized_updates": serialized, "policies": sorted(policies),
            "maximum_action_error": maximum_error,
            "input_sha256": {"samples.json": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
                             "events.json": hashlib.sha256(events_path.read_bytes()).hexdigest()},
            "gpu_acceptance_performed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--policy-registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory.resolve(strict=True), args.catalog.resolve(strict=True), args.policy_registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
