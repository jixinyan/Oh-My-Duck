import argparse
import asyncio
import importlib.metadata
import json
from pathlib import Path

import viser
from websockets.asyncio.client import connect

from accept_harness_motion_guard import available_port


async def run(output):
    port = available_port()
    server = viser.ViserServer(host="127.0.0.1", port=port)
    try:
        server.scene.add_frame("/transport-acceptance")
        async with connect(f"ws://127.0.0.1:{port}", subprotocols=[f"viser-v{viser.__version__}"]) as connection:
            async with asyncio.timeout(15):
                packet = await connection.recv()
            if not isinstance(packet, bytes) or not packet:
                raise AssertionError("Viser websocket did not deliver native binary data")
        result = {"accepted": True, "received_packet_bytes": len(packet),
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("websockets", "viser", "jsonschema", "stable-baselines3")}}
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Transport evidence output must be new")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(args.output))


if __name__ == "__main__":
    main()
