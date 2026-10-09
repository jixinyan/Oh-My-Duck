import json
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from oh_my_duck.perception.frames import decode_frame, validate_prompt
from oh_my_duck.perception.validation import validate_measurements, validate_response


class PerceptionClient:
    def __init__(self, endpoint: str):
        if not isinstance(endpoint, str) or not endpoint or any(character.isspace() for character in endpoint):
            raise ValueError("Perception endpoint must be a nonempty URL string")
        address = urlsplit(endpoint)
        if (address.scheme != "http" or address.hostname != "127.0.0.1" or address.port is None
                or not 1 <= address.port <= 65535 or address.username is not None or address.password is not None
                or address.path not in ("", "/") or address.query or address.fragment):
            raise ValueError("Perception endpoint must be an explicit local service or SSH tunnel")
        self.endpoint = endpoint.rstrip("/")

    def inspect(self, frame: dict, prompt: str) -> dict:
        validate_prompt(prompt)
        decoded = decode_frame(frame)
        request = Request(self.endpoint + "/inspect", data=json.dumps({"frame": frame, "prompt": prompt},
                          allow_nan=False).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        validate_response(frame, prompt, result)
        validate_measurements(frame, result, decoded=decoded)
        return result
