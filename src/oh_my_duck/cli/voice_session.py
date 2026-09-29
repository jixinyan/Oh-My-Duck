import argparse
import asyncio
from pathlib import Path

from oh_my_duck.core.contracts.identity import ExecutionDomain
from oh_my_duck.voice.device import PortAudioDevice
from oh_my_duck.voice.remote import RemoteVoiceServices
from oh_my_duck.voice.session import VoiceSession


def main() -> int:
    parser = argparse.ArgumentParser(description="显式录音与播音 JSONL 会话")
    parser.add_argument("--persona", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--domain", choices=[domain.value for domain in ExecutionDomain], required=True)
    parser.add_argument("--input-device", required=True)
    parser.add_argument("--output-device", required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--asr-url", default="http://127.0.0.1:18761")
    parser.add_argument("--tts-url", default="http://127.0.0.1:18762")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    device = PortAudioDevice(
        args.data_dir / "recordings", input_device=args.input_device,
        output_device=args.output_device, sample_rate=args.sample_rate,
    )
    services = RemoteVoiceServices(args.asr_url, args.tts_url, args.data_dir / "speech")
    try:
        asyncio.run(VoiceSession(
            device, services, persona_id=args.persona, robot_id=args.robot_id,
            domain=ExecutionDomain(args.domain), log_path=args.log,
        ).run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
