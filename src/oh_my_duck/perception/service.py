import argparse
import base64
import hashlib
from io import BytesIO
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel
import torch
from ultralytics import YOLO
import uvicorn

from oh_my_duck.perception.rgbd import measure_target


class InspectRequest(BaseModel):
    frame: dict
    prompt: str


class PerceptionModels:
    def __init__(self, yolo_path: Path, sam_path: Path | None, output: Path):
        self.yolo_path = yolo_path.resolve(strict=True)
        self.yolo = YOLO(str(self.yolo_path))
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.sam = None
        self.lock = Lock()
        self.model_hashes = {"yolo26": hashlib.sha256(self.yolo_path.read_bytes()).hexdigest()}
        if sam_path is not None:
            from oh_my_duck.perception.sam31 import load_predictor
            checkpoint = sam_path.resolve(strict=True)
            self.sam = load_predictor(checkpoint)
            self.model_hashes["sam3.1"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    @torch.inference_mode()
    def inspect(self, frame: dict, prompt: str) -> dict:
        if not prompt.strip() or len(prompt) > 120:
            raise ValueError("Prompt must contain 1–120 characters")
        pixels = base64.b64decode(frame["rgb_png_base64"], validate=True)
        image = Image.open(BytesIO(pixels)).convert("RGB")
        points = np.load(BytesIO(base64.b64decode(frame["points_world_npy_base64"], validate=True)), allow_pickle=False)
        if points.shape != (image.height, image.width, 3):
            raise ValueError("RGB and depth geometry differ")
        predictions = self.yolo.predict(image, device=0, conf=0.15, verbose=False)[0]
        boxes = predictions.boxes.xyxy.cpu().numpy()
        scores = predictions.boxes.conf.cpu().numpy()
        classes = predictions.boxes.cls.cpu().numpy().astype(int)
        masks, sam_scores = [], []
        if self.sam is not None:
            directory = self.output / uuid4().hex
            directory.mkdir()
            image.save(directory / "000000.jpg", quality=100, subsampling=0)
            session = self.sam.handle_request({"type": "start_session", "resource_path": str(directory)})["session_id"]
            try:
                response = self.sam.handle_request({"type": "add_prompt", "session_id": session,
                    "frame_index": 0, "text": prompt})["outputs"]
                masks = np.asarray(response["out_binary_masks"], dtype=bool)
                sam_scores = np.asarray(response["out_probs"])
            finally:
                self.sam.handle_request({"type": "close_session", "session_id": session})
        targets = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        candidates = list(range(len(masks))) if self.sam is not None else list(range(len(boxes)))
        for index in candidates:
            if self.sam is not None:
                mask = masks[index]
                rows, columns = np.nonzero(mask)
                if len(rows) < 8:
                    continue
                box = np.array([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1])
                intersections = np.maximum(0, np.minimum(boxes[:, 2:], box[2:]) - np.maximum(boxes[:, :2], box[:2])).prod(axis=-1)
                areas = (boxes[:, 2:] - boxes[:, :2]).prod(axis=-1) + (box[2:] - box[:2]).prod() - intersections
                matches = intersections / np.maximum(areas, 1)
                match = int(matches.argmax()) if len(matches) and matches.max() >= 0.1 else None
                label, confidence = prompt, float(sam_scores[index])
            else:
                box, confidence = boxes[index], float(scores[index])
                label, match = predictions.names[int(classes[index])], index
                if prompt != "objects" and prompt.lower() not in label.lower():
                    continue
                mask = np.zeros((image.height, image.width), dtype=bool)
                left, top, right, bottom = np.rint(box).astype(int)
                mask[max(top, 0):min(bottom, image.height), max(left, 0):min(right, image.width)] = True
            measured = measure_target(mask, points, frame["camera_position_m"], frame["body_position_m"], frame["yaw_rad"])
            target = {"target_id": f"{frame['episode_id']}:{frame['sequence']}:{index}",
                      "label": label, "confidence": confidence, "bbox_xyxy": box.tolist(),
                      "detection_source": "sam3.1" if self.sam is not None else "yolo26",
                      "mask_source": "sam3.1" if self.sam is not None else "yolo26_bbox",
                      "distance_source": frame["distance_source"], **measured,
                      "yolo_match": None if match is None else {"label": predictions.names[int(classes[match])],
                                                               "confidence": float(scores[match])}}
            targets.append(target)
            if self.sam is not None:
                overlay = np.asarray(annotated).copy()
                overlay[mask] = (overlay[mask].astype(np.uint16) * 2 +
                                 np.asarray([30, 220, 120], dtype=np.uint16)) // 3
                annotated = Image.fromarray(overlay)
                draw = ImageDraw.Draw(annotated)
            draw.rectangle(box.tolist(), outline="lime", width=2)
            draw.text((float(box[0]), float(box[1])), label, fill="white")
        output = BytesIO()
        annotated.save(output, format="PNG")
        return {"episode_id": frame["episode_id"], "sequence": frame["sequence"],
                "observed_at": frame["observed_at"], "prompt": prompt, "targets": targets,
                "detection_source": "sam3.1" if self.sam is not None else "yolo26",
                "models": self.model_hashes, "distance_source": frame["distance_source"],
                "rgb_png_base64": base64.b64encode(output.getvalue()).decode(),
                "image_sha256": hashlib.sha256(pixels).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", type=Path, required=True)
    parser.add_argument("--sam", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8784)
    args = parser.parse_args()
    models = PerceptionModels(args.yolo, args.sam, args.output)
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"models": models.model_hashes, "engine": "yolo26-sam31" if models.sam is not None else "yolo26"}

    @app.post("/inspect")
    def inspect(request: InspectRequest):
        with models.lock:
            return models.inspect(request.frame, request.prompt)

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
