import json
from urllib.request import Request, urlopen


class PerceptionClient:
    def __init__(self, endpoint: str):
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError("Perception endpoint must be an explicit local service or SSH tunnel")
        self.endpoint = endpoint.rstrip("/")

    def inspect(self, frame: dict, prompt: str) -> dict:
        request = Request(self.endpoint + "/inspect", data=json.dumps({"frame": frame, "prompt": prompt},
                          allow_nan=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result["episode_id"] != frame["episode_id"] or result["sequence"] != frame["sequence"]:
            raise RuntimeError("Perception response refers to a different physical frame")
        return result
