import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate registered policy packages with actual CPU ONNX inference")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from oh_my_duck.robotics.policies.catalogue import PolicyCatalogue

    catalogue = PolicyCatalogue(args.catalog, args.registry)
    result = {"passed": True, "revision": catalogue.revision, "policies": catalogue.describe_apartment(),
              "locomotion": {model: catalogue.locomotion_policy(model)
                             for model in ("allcollisions", "groundcontact_rollers")},
              "scope": "Registered package identity, metadata and CPU inference",
              "behavior_acceptance_performed": False, "gpu_acceptance_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "revision": catalogue.revision, "policies": len(catalogue.policies)}))


if __name__ == "__main__":
    main()
