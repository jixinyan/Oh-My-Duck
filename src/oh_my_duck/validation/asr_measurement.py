import argparse
import asyncio
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import time

from oh_my_duck.validation.asr_corpus import calculate_metrics, file_sha256, normalize_text
from oh_my_duck.validation.asr_preparation import revision_argument


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("CPU threads 必须是正整数")
    return number


def validate_preparation(path: Path) -> dict:
    import pyarrow.parquet as parquet

    prepared = json.loads(path.read_text())
    if prepared["passed"] is not True or prepared["dataset"] != "google/fleurs" or prepared["split"] != "test":
        raise ValueError("测量需要已通过准备检查的 FLEURS test 语料")
    if not prepared["splits"] or type(prepared["total_rows"]) is not int or prepared["total_rows"] <= 0:
        raise ValueError("语料必须包含实际录音")
    revision_argument(prepared["dataset_revision"])
    for language, definition in prepared["splits"].items():
        if re.fullmatch(r"[a-z]{2,3}_[a-z0-9_]+", language) is None:
            raise ValueError("语料语言必须使用 FLEURS config 标识")
        source = Path(definition["path"]).resolve(strict=True)
        if type(definition["rows"]) is not int or definition["rows"] <= 0 or (
            parquet.ParquetFile(source).metadata.num_rows != definition["rows"]
        ):
            raise ValueError("测量必须使用原始语料文件的完整数量")
        if file_sha256(source) != definition["sha256"]:
            raise ValueError("原始语料文件 SHA256 不一致")
    if sum(item["rows"] for item in prepared["splits"].values()) != prepared["total_rows"]:
        raise ValueError("语料总数与逐项数量不一致")
    return prepared


async def transcribe_corpus(recognition, prepared: dict, output: Path) -> dict:
    import numpy as np
    import pyarrow.parquet as parquet
    import soundfile as sf

    from oh_my_duck.core.contracts.sensors import PayloadRef

    audio_directory = output / "audio"
    audio_directory.mkdir()
    summaries = {}
    with (output / "predictions.jsonl").open("x") as predictions:
        for language, definition in prepared["splits"].items():
            records = []
            for index, batch in enumerate(parquet.ParquetFile(definition["path"]).iter_batches(batch_size=1)):
                original = batch.to_pylist()[0]
                audio, sample_rate = sf.read(BytesIO(original["audio"]["bytes"]),
                                            dtype="float32", always_2d=True)
                if not audio.size or not np.isfinite(audio).all():
                    raise ValueError("原始录音必须包含有限采样值")
                audio_path = audio_directory / f"{language}-{index:04d}.wav"
                with audio_path.open("xb") as file:
                    sf.write(file, audio, sample_rate, format="WAV", subtype="FLOAT")
                decoded, actual_rate = sf.read(audio_path, dtype="float32", always_2d=True)
                if sample_rate != actual_rate or not np.array_equal(audio, decoded):
                    raise ValueError("测量 WAV 的采样值与原始录音不一致")
                reference = original["transcription"]
                if not isinstance(reference, str) or not normalize_text(reference):
                    raise ValueError("原始录音必须具有有效的参考文字")
                audio_sha = file_sha256(audio_path)
                began = time.perf_counter()
                hypothesis = await recognition.transcribe(PayloadRef(
                    uri=audio_path.as_uri(), media_type="audio/wav", sha256=audio_sha,
                ))
                elapsed = time.perf_counter() - began
                record = {
                    "language": language, "row_index": index, "source_id": original["id"],
                    "source_audio_path": original["audio"]["path"],
                    "source_audio_sha256": hashlib.sha256(original["audio"]["bytes"]).hexdigest(),
                    "audio_path": str(audio_path), "audio_sha256": audio_sha,
                    "sample_rate": sample_rate, "frames": len(audio), "channels": audio.shape[1],
                    "decoded_samples_equal": True, "reference": reference,
                    "raw_reference": original["raw_transcription"], "hypothesis": hypothesis,
                    "normalized_reference": normalize_text(reference),
                    "normalized_hypothesis": normalize_text(hypothesis),
                    "duration_s": len(audio) / sample_rate, "transcribe_seconds": elapsed,
                }
                records.append(record)
                predictions.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                predictions.flush()
                if len(records) % 25 == 0:
                    print(json.dumps({"state": "asr_progress", "language": language,
                                      "records": len(records), "total_records": definition["rows"]}), flush=True)
            if len(records) != definition["rows"]:
                raise ValueError("转写必须覆盖所选语料的全部录音")
            summaries[language] = calculate_metrics(records, english=language == "en_us")
            print(json.dumps({"state": "language_completed", "language": language,
                              "measurements": summaries[language]}), flush=True)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="使用项目 Qwen ASR 在 CPU 转写全部真实录音并保存识别指标")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-revision", type=revision_argument, required=True)
    parser.add_argument("--cpu-threads", type=positive_integer, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU 语料测量要求 CUDA_VISIBLE_DEVICES 为空")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source = args.source_root.expanduser().resolve(strict=True)
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
                                    text=True)
    if dirty:
        raise ValueError("测量源码必须使用没有 tracked 修改的固定 Git 目录")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if file_sha256(Path(__file__)) != file_sha256(source / "src/oh_my_duck/validation/asr_measurement.py"):
        raise ValueError("实际测量程序与声明源码不一致")
    corpus = args.corpus.expanduser().resolve(strict=True)
    prepared = validate_preparation(corpus)

    import torch
    from oh_my_duck.voice.qwen import ASR_MODEL, QwenSpeechRecognition
    import oh_my_duck.voice.qwen as qwen_module

    if torch.cuda.is_initialized() or file_sha256(Path(qwen_module.__file__)) != file_sha256(
        source / "src/oh_my_duck/voice/qwen.py"
    ):
        raise ValueError("CPU 模型状态或实际 Qwen 源码不一致")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    recognition = QwenSpeechRecognition(device="cpu", revision=args.model_revision)
    load_seconds = time.perf_counter() - started
    if recognition.model_revision != f"{ASR_MODEL}@{args.model_revision}":
        raise ValueError("实际 ASR 模型与声明 revision 不一致")
    parameters = list(recognition.model.model.parameters())
    if not parameters or any(parameter.device.type != "cpu" or parameter.dtype != torch.float32
                             for parameter in parameters):
        raise ValueError("语料测量要求全部模型参数为 CPU Float32")
    summaries = asyncio.run(transcribe_corpus(recognition, prepared, output))
    if torch.cuda.is_initialized() or sum(item["records"] for item in summaries.values()) != prepared["total_rows"]:
        raise ValueError("CPU 测量状态或完整录音数量不一致")
    result = {
        "completed": True, "source_revision": revision, "dataset": prepared["dataset"],
        "dataset_revision": prepared["dataset_revision"], "split": prepared["split"],
        "model_revision": recognition.model_revision, "device": "cpu", "dtype": "torch.float32",
        "model_load_seconds": load_seconds, "threads": torch.get_num_threads(),
        "max_new_tokens": 512, "max_inference_batch_size": 1, "language_hint": None,
        "reference_field": "transcription", "accuracy_threshold_declared": False,
        "normalization": "Unicode NFKC, casefold, Unicode punctuation removal, whitespace collapse; CER removes whitespace",
        "official_benchmark_reproduction": False, "preparation_sha256": file_sha256(corpus),
        "predictions_sha256": file_sha256(output / "predictions.jsonl"),
        "runner_sha256": file_sha256(Path(__file__)), "qwen_module_sha256": file_sha256(Path(qwen_module.__file__)),
        "measurements": summaries, "cuda_initialized": False, "gpu_acceptance_performed": False,
    }
    with (output / "result.json").open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps({"completed": True, "test_records": prepared["total_rows"],
                      "gpu_acceptance_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
