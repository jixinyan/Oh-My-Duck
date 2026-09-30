import argparse
import json
from pathlib import Path

import numpy as np

from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend, MotionBusyError
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def run() -> dict:
    root = Path(__file__).resolve().parents[1]
    catalog = root / ".cache" / "official-policies" / OFFICIAL_REVISION
    backend = CpuMujocoBamBackend(robot_id="official-backend-contract", catalog_dir=catalog)
    try:
        initial = backend.reset_episode(seed=42, goal={"kind": "room", "room": "corridor", "hold_ticks": 5})
        same_a = backend.observe_control()
        same_b = backend.observe_control()
        if any(item["sequence"] != 0 or item["simulation_time_s"] != 0 for item in (initial, same_a, same_b)):
            raise ValueError("Reading observations advanced native physics")
        if backend._stopped_samples != 0 or backend._goal_held_ticks != 0:
            raise ValueError("Observation reads changed temporal evidence")
        first_check = backend.check_goal()
        second_check = backend.check_goal()
        if first_check["complete"] or second_check["complete"] or backend._goal_held_ticks != 0:
            raise ValueError("Formal goal checks changed continuous native hold evidence")
        first_action = None
        for index in range(50):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(inference["action"], request_id=f"stand:{index}",
                                                 expected_sequence=inference["sequence"],
                                                 should_stop=lambda: False)
            if index == 0:
                first_action = (inference, result)
        standing_goal = backend.check_goal()
        if not standing_goal["complete"] or standing_goal["checks"]["goal_reached"]["evidence"]["held_ticks"] != 50:
            raise ValueError("Room GT did not accumulate one hold sample per physical control action")
        for _ in range(3):
            backend.observe_control()
            backend.check_goal()
        if backend._goal_held_ticks != 50 or backend._stopped_samples < 5:
            raise ValueError("Repeated reads changed stop or goal evidence")
        initial_receipt = first_action[1]
        replay_receipt = backend.apply_policy_action(first_action[0]["action"], request_id="stand:0",
                                                     expected_sequence=first_action[0]["sequence"])
        if replay_receipt != initial_receipt or backend._sequence != 50:
            raise ValueError("Duplicate executed request advanced native physics")
        backend.bind_goal({"kind": "object", "object": "obj_3", "distance_m": 0.2, "hold_ticks": 5})
        object_check = backend.check_goal()
        if object_check["complete"] or object_check["checks"]["goal_reached"]["evidence"]["target"]["body"] != "obj_3":
            raise ValueError("Native object-distance GT failed")
        backend.bind_goal({"kind": "dock", "hold_ticks": 5})
        dock_check = backend.check_goal()
        if dock_check["complete"] or dock_check["checks"]["goal_reached"]["evidence"]["target"]["site"] != "dock_site":
            raise ValueError("Native dock-site GT failed")
        inference = backend.infer_policy()
        discarded = backend.discard_pending_inference()
        if not discarded["discarded"] or not discarded["confirmed_stopped"]:
            raise ValueError("Confirmed stopped boundary failed to discard stale inference")
        selected = backend.select_policy("sitstand", request_id="select:sitstand")
        if selected["action_spec"]["version"] != 1:
            raise ValueError("Policy selection changed the action specification version")
        backend.set_command({"posture": "sit"}, request_id="posture:sit")
        for index in range(100):
            inference = backend.infer_policy()
            if inference["command_block"][0] != 1.0:
                raise ValueError("Sit posture flag did not enter official observation")
            result = backend.apply_policy_action(inference["action"], request_id=f"sit:{index}",
                                                 expected_sequence=inference["sequence"],
                                                 should_stop=lambda: False)
        if result["measurements"]["height_m"] >= 0.08 or result["measurements"]["episode_terminated"]:
            raise ValueError("Scripted sitting pose was interrupted")
        backend.bind_goal({"kind": "room", "room": "corridor", "hold_ticks": 5})
        seated_goal = backend.check_goal()
        if seated_goal["complete"]:
            raise ValueError("A seated robot satisfied the standing navigation goal")
        pending = backend.infer_policy()
        targets_before_stop = backend.controller.q_target.copy()
        rejected = False
        try:
            backend.apply_policy_action(pending["action"], request_id="stop:before-target",
                                        expected_sequence=pending["sequence"], should_stop=lambda: True)
        except MotionBusyError:
            rejected = True
        if not rejected or not np.array_equal(targets_before_stop, backend.controller.q_target):
            raise ValueError("Stop request changed targets before actuation")
        backend.set_command({"posture": "stand"}, request_id="posture:stand")
        pending = backend.infer_policy()
        stop_calls = 0

        def stop_after_two_steps() -> bool:
            nonlocal stop_calls
            stop_calls += 1
            return stop_calls >= 4

        partial = backend.apply_policy_action(pending["action"], request_id="stop:partial",
                                              expected_sequence=pending["sequence"],
                                              should_stop=stop_after_two_steps)
        if partial["raw_sim_steps"] != 2 or not partial["interrupted"]:
            raise ValueError("Physical stop did not interrupt after two substeps")
        replay_rejected = False
        try:
            backend.apply_policy_action(pending["action"], request_id="stop:partial",
                                        expected_sequence=pending["sequence"])
        except ValueError:
            replay_rejected = True
        if not replay_rejected:
            raise ValueError("Interrupted action request was replayed")
        pending = backend.infer_policy()
        early_discard = backend.discard_pending_inference()
        if not early_discard["discarded"] or early_discard["confirmed_stopped"]:
            raise ValueError("Paused pending inference failed to clear before stop confirmation")
        return {"status": "passed", "episode_id": backend.episode_id,
                "observation_read_sequence": same_b["sequence"],
                "standing_hold_ticks": standing_goal["checks"]["goal_reached"]["evidence"]["held_ticks"],
                "object_target": object_check["checks"]["goal_reached"]["evidence"]["target"],
                "dock_target": dock_check["checks"]["goal_reached"]["evidence"]["target"],
                "seated_goal_complete": seated_goal["complete"],
                "seated_height_m": result["measurements"]["height_m"],
                "pre_actuation_stop_rejected": rejected,
                "partial_raw_sim_steps": partial["raw_sim_steps"],
                "partial_request_replay_rejected": replay_rejected,
                "early_pending_discard": early_discard,
                "final_sequence": backend._sequence,
                "simulation_time_s": float(backend.data.time)}
    finally:
        backend.close()


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
