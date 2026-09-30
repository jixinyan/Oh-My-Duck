import argparse
import json
from pathlib import Path

from diagnose_apartment_collision import contact_evidence, snapshot
from diagnose_apartment_diagonals import GROUND_GEOMS, concise
from oh_my_duck.robotics.backends.simulation import CpuMujocoBamBackend
from oh_my_duck.robotics.microduck.official_policies import OFFICIAL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Diagnostic output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    backend = CpuMujocoBamBackend(robot_id="after-goal-diagnosis",
                                  catalog_dir=root / ".cache" / "official-policies" / OFFICIAL_REVISION)
    traces = []
    first_goal = None
    first_non_ground = None
    try:
        first = backend.reset_episode(seed=20260929,
                                      goal={"kind": "room", "room": "office", "hold_ticks": 5})
        traces.append(concise(snapshot(backend, first["measurements"], "initial")))
        for index in range(187):
            inference = backend.infer_policy()
            result = backend.apply_policy_action(
                inference["action"], request_id=f"after-goal:velstand:{index}",
                expected_sequence=inference["sequence"], should_stop=lambda: False)
        backend.select_policy("alpha_walking", request_id="after-goal:select")
        for phase, count, command in (("diagonal", 600, [0.2, 0.25, 0.0]),
                                      ("stop", 50, [0.0, 0.0, 0.0])):
            backend.set_command({"twist": command}, request_id=f"after-goal:command:{phase}")
            for index in range(count):
                inference = backend.infer_policy()
                result = backend.apply_policy_action(
                    inference["action"], request_id=f"after-goal:{phase}:{index}",
                    expected_sequence=inference["sequence"], should_stop=lambda: False)
                contacts = contact_evidence(backend)
                non_ground = {geom: value for geom, value in contacts["external_geom_counts"].items()
                              if geom not in GROUND_GEOMS}
                if non_ground and first_non_ground is None:
                    first_non_ground = {"sequence": result["sequence"], "geoms": non_ground,
                                        "position_m": result["measurements"]["body_position_m"]}
                goal = backend.check_goal()["checks"]["goal_reached"]["satisfied"]
                if goal and first_goal is None:
                    first_goal = {"sequence": result["sequence"],
                                  "position_m": result["measurements"]["body_position_m"]}
                if result["sequence"] % 25 == 0 or index == count - 1 or non_ground:
                    traces.append(concise(snapshot(backend, result["measurements"], phase)))
                if result["measurements"]["fallen"]:
                    raise RuntimeError(f"Robot fell at sequence {result['sequence']}")
        final_goal = backend.check_goal()
    finally:
        backend.close()
    output = {"seed": 20260929, "goal": {"kind": "room", "room": "office", "hold_ticks": 5},
              "phases": [{"name": "velstand", "controls": 187, "twist": [0, 0, 0]},
                         {"name": "diagonal", "controls": 600, "twist": [0.2, 0.25, 0]},
                         {"name": "stop", "controls": 50, "twist": [0, 0, 0]}],
              "first_goal": first_goal, "first_non_ground_contact": first_non_ground,
              "final_goal": final_goal, "traces": traces}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in ("first_goal", "first_non_ground_contact",
                                                   "final_goal")}, indent=2))


if __name__ == "__main__":
    main()
