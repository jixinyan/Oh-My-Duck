import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                         "active_gpu": args.gpu, "physics_gpu": args.gpu, "multi_gpu": False})
    try:
        import omni.replicator.core as rep
        import numpy as np
        from PIL import Image

        rep.create.light(light_type="Dome", intensity=1000)
        rep.create.cube(position=(0, 0, 0))
        camera = rep.create.camera(position=(3, 3, 3), look_at=(0, 0, 0))
        product = rep.create.render_product(camera, (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        for _ in range(30):
            rep.orchestrator.step(rt_subframes=4)
        pixels = np.asarray(rgb.get_data())
        if pixels.shape != (480, 640, 4) or pixels.dtype != np.uint8 or float(pixels.std()) < 1:
            raise RuntimeError(f"RTX produced invalid image: {pixels.shape}")
        Image.fromarray(pixels).save(args.output / "rtx-device.png")
        (args.output / "result.json").write_text(json.dumps({"scope": "actual RTX rendering device availability",
            "rgb_std": float(pixels[..., :3].std()), "resolution": [640, 480]}) + "\n")
    finally:
        app.close()


if __name__ == "__main__":
    main()
