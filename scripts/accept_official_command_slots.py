import argparse
import json
from pathlib import Path

import numpy as np

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION, OfficialPolicyCatalogue


def run() -> dict:
    root = Path(__file__).resolve().parents[1]
    catalogue = OfficialPolicyCatalogue(root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    metadata = {item["name"]: item for item in catalogue.describe_apartment()}
    sitstand = metadata["sitstand"]
    if sitstand["ramp_s"] != 2.0 or sitstand["unwind_s"] != 1.0:
        raise ValueError("Sitstand timing differs from the pinned manifest")
    if metadata["velstand"]["slot"] != "walk" or metadata["velstand"]["entry_pose"] != "standing":
        raise ValueError("Walk entry metadata differs from the pinned manifest")
    if metadata["roulade"]["chain"] is not True:
        raise ValueError("Roulade chain metadata differs from the pinned manifest")
    rejected = {}
    allowed = {}
    zero3, zero4, zero6 = (0.0,) * 3, (0.0,) * 4, (0.0,) * 6
    for name, policy in catalogue.policies.items():
        block = catalogue.command_block(policy, velocity=zero3, posture="stand", elapsed_s=0,
                                        head=zero4, body=zero6)
        if block.shape != (13,) or not np.isfinite(block).all():
            raise ValueError(f"Invalid official command block: {name}")
        channels = metadata[name]["command_channels"]
        if name in {"roulade", "kick_left", "kick_right"} and np.any(block):
            raise ValueError(f"Behavior policy received a nonzero command: {name}")
        rejected[name] = []
        for channel, values in (("twist", (0.1, 0.0, 0.0)),
                                ("head", (0.1, 0.0, 0.0, 0.0)),
                                ("body", (0.0, 0.0, 0.01, 0.0, 0.0, 0.0))):
            arguments = {"velocity": zero3, "posture": "stand", "elapsed_s": 0,
                         "head": zero4, "body": zero6}
            arguments[{"twist": "velocity", "head": "head", "body": "body"}[channel]] = values
            if channel in channels:
                filled = catalogue.command_block(policy, **arguments)
                if not np.any(filled):
                    raise ValueError(f"Supported command did not enter the observation: {name}:{channel}")
                allowed.setdefault(name, []).append(channel)
            else:
                try:
                    catalogue.command_block(policy, **arguments)
                except ValueError:
                    rejected[name].append(channel)
                else:
                    raise ValueError(f"Unsupported command was accepted: {name}:{channel}")
    backend = CpuMujocoBamBackend(robot_id="official-command-contract", catalog_dir=catalogue.directory)
    try:
        backend.reset_episode(seed=42, goal={"kind": "room", "room": "corridor", "hold_ticks": 5})
        for index in range(50):
            inference = backend.infer_policy()
            backend.apply_policy_action(inference["action"], request_id=f"stand:{index}",
                                        expected_sequence=inference["sequence"], should_stop=lambda: False)
        backend.select_policy("sitstand", request_id="select:sitstand")
        backend.set_command({"head": [0.0, 0.2, 0.0, 0.0], "posture": "sit"}, request_id="sit:head")
        sit_inference = backend.infer_policy()
        if sit_inference["command_block"][0] != 1.0 or not np.isclose(sit_inference["command_block"][4], 0.2):
            raise ValueError("Sitstand head and posture commands did not enter the observation")
        unsupported_rejected = False
        try:
            backend.set_command({"body": [0.0, 0.0, 0.01, 0.0, 0.0, 0.0]}, request_id="sit:body")
        except ValueError:
            unsupported_rejected = True
        if not unsupported_rejected:
            raise ValueError("Sitstand accepted a body command")
        backend.discard_pending_inference()
    finally:
        backend.close()
    return {"status": "passed", "manifest_revision": OFFICIAL_REVISION,
            "sitstand_timing": {"ramp_s": sitstand["ramp_s"], "unwind_s": sitstand["unwind_s"]},
            "allowed_channels": allowed, "rejected_channels": rejected,
            "sitstand_body_rejected": unsupported_rejected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
