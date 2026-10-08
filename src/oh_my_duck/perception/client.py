import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from oh_my_duck.perception.validation import validate_response


class PerceptionClient:
    def __init__(self, endpoint: str):
        address = urlsplit(endpoint)
        if (address.scheme != "http" or address.hostname != "127.0.0.1" or address.port is None
                or not 1 <= address.port <= 65535 or address.username is not None or address.password is not None
                or address.path not in ("", "/") or address.query or address.fragment):
            raise ValueError("Perception endpoint must be an explicit local service or SSH tunnel")
        self.endpoint = endpoint.rstrip("/")

    def inspect(self, frame: dict, prompt: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 120:
            raise ValueError("Prompt must contain 1–120 characters")
        request = Request(self.endpoint + "/inspect", data=json.dumps({"frame": frame, "prompt": prompt},
                          allow_nan=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        validate_response(frame, prompt, result)
        return result
