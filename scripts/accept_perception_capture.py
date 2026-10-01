import argparse
import base64
import json
from pathlib import Path

from oh_my_duck.perception.client import PerceptionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    client = PerceptionClient(args.endpoint)
    results = []
    for metadata in sorted(args.capture.glob("frame-*.json")):
        index = metadata.stem.removeprefix("frame-")
        frame = json.loads(metadata.read_text())
        frame["rgb_png_base64"] = base64.b64encode((args.capture / f"head-{index}.png").read_bytes()).decode()
        frame["points_world_npy_base64"] = base64.b64encode((args.capture / f"points-{index}.npy").read_bytes()).decode()
        for prompt_index, prompt in enumerate(args.prompts):
            result = client.inspect(frame, prompt)
            image = result.pop("rgb_png_base64")
            (args.output / f"model-{index}-{prompt_index}.png").write_bytes(base64.b64decode(image, validate=True))
            results.append(result)
            (args.output / "result.json").write_text(json.dumps({"scope": "real model inference on recorded Newton RGBD",
                                                                "results": results}, indent=2) + "\n")
            print(json.dumps({"frame": index, "prompt": prompt, "targets": len(result["targets"])}), flush=True)
    if not results or not any(target["distance_status"] == "valid" for result in results for target in result["targets"]):
        raise AssertionError("Recorded frames contain no valid model-derived target distance")


if __name__ == "__main__":
    main()
