import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-sam", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    yolo = args.output / "yolo26s.pt"
    YOLO(str(yolo))
    report = {"yolo_checkpoint": str(yolo.resolve()), "yolo_sha256": hashlib.sha256(yolo.read_bytes()).hexdigest()}
    if args.include_sam:
        metadata = HfApi().model_info("facebook/sam3.1", token=True)
        checkpoint = Path(hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt",
                                         revision=metadata.sha, token=True, local_dir=args.output / "sam3.1"))
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        report.update(sam_revision=metadata.sha, sam_checkpoint=str(checkpoint.resolve()), sam_sha256=digest)
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
