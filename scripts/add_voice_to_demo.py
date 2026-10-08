import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit
import wave

import imageio_ffmpeg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="将实际指令音频和固定音色反馈写入 agentic MP4")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--voice-result", type=Path, required=True)
    parser.add_argument("--instruction-audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".voice.json").exists():
        raise FileExistsError(args.output)
    result = json.loads(args.voice_result.read_text())
    if result["native_run"]["state"] != "succeeded" or not any(
            verdict["status"] == "passed" for verdict in result["native_run"]["verdicts"]):
        raise ValueError("语音任务没有通过原生验证")
    video_report_path = args.video.with_suffix(".json")
    video_report = json.loads(video_report_path.read_text())
    if (video_report["runId"] != result["native_run"]["id"] or
            video_report["runState"] != "succeeded" or video_report["formalVerdict"] != "passed" or
            video_report["videoSha256"] != digest(args.video)):
        raise ValueError("视频来源与语音任务记录不一致")
    response = Path(unquote(urlsplit(result["speech"]["audio"]["uri"]).path)).resolve(strict=True)
    if (digest(args.instruction_audio) != result["transcription"]["audio_sha256"] or
            digest(response) != result["speech"]["audio"]["sha256"]):
        raise ValueError("音频与任务记录 SHA256 不一致")
    frames = imageio_ffmpeg.read_frames(str(args.video), pix_fmt="rgb24")
    metadata = next(frames)
    frames.close()
    with wave.open(str(response)) as audio:
        response_duration = audio.getnframes() / audio.getframerate()
    with wave.open(str(args.instruction_audio)) as audio:
        instruction_duration = audio.getnframes() / audio.getframerate()
    duration = metadata["duration"]
    if (not math.isfinite(duration) or duration <= 0 or instruction_duration <= 0 or
            response_duration <= 0 or instruction_duration + response_duration + 0.5 > duration):
        raise ValueError("视频时长无法完整容纳指令音频和反馈音频")
    offset_ms = round((duration - response_duration - 0.5) * 1000)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-i", str(args.video),
        "-i", str(args.instruction_audio), "-i", str(response), "-filter_complex",
        f"[2:a]adelay={offset_ms}:all=1[feedback];[1:a][feedback]amix=inputs=2:duration=longest:normalize=0,apad[audio]",
        "-map", "0:v:0", "-map", "[audio]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(duration), "-movflags", "+faststart", str(args.output)]
    subprocess.run(command, check=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(args.output),
        "-f", "null", "-"], check=True, capture_output=True)
    manifest = {"scope": result["scope"], "run_id": result["native_run"]["id"],
        "transcription": result["transcription"], "speech": result["speech"],
        "instruction_audio_sha256": digest(args.instruction_audio), "response_offset_ms": offset_ms,
        "video_source_sha256": digest(args.video), "video_sha256": digest(args.output),
        "video_report_sha256": digest(video_report_path), "voice_result_sha256": digest(args.voice_result),
        "instruction_duration_s": instruction_duration, "response_duration_s": response_duration,
        "duration_s": duration, "complete_decode": "passed",
        "live_microphone": result["live_microphone"], "speaker_playback": result["speaker_playback"]}
    args.output.with_suffix(".voice.json").write_text(json.dumps(manifest, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
