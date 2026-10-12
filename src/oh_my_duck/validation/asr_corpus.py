import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import subprocess
import unicodedata


def file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize_text(value: str) -> str:
    import regex

    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(regex.sub(r"\p{P}", "", value).split())


def calculate_metrics(records: list[dict], *, english: bool) -> dict:
    import jiwer
    import numpy as np

    references = [normalize_text(record["reference"]) for record in records]
    hypotheses = [normalize_text(record["hypothesis"]) for record in records]
    if not records or not all(references):
        raise ValueError("ASR 语料必须包含有效参考文字")
    characters = jiwer.process_characters(
        ["".join(value.split()) for value in references],
        ["".join(value.split()) for value in hypotheses],
    )
    elapsed = [record["transcribe_seconds"] for record in records]
    durations = [record["duration_s"] for record in records]
    if not all(type(value) in (int, float) and math.isfinite(value) and value > 0
               for value in [*elapsed, *durations]):
        raise ValueError("音频时长和推理耗时必须为正数且有限")
    result = {
        "records": len(records), "character_error_rate": characters.cer,
        "character_counts": {name: getattr(characters, name)
                             for name in ("hits", "substitutions", "deletions", "insertions")},
        "audio_seconds": sum(durations), "transcribe_seconds": sum(elapsed),
        "real_time_factor": sum(elapsed) / sum(durations),
        "median_transcribe_seconds": statistics.median(elapsed),
        "p95_transcribe_seconds": float(np.percentile(elapsed, 95)),
    }
    if english:
        words = jiwer.process_words(references, hypotheses)
        result["word_error_rate"] = words.wer
        result["word_counts"] = {name: getattr(words, name)
                                 for name in ("hits", "substitutions", "deletions", "insertions")}
    return result


def audit_corpus(capture: Path, corpus: Path, source_root: Path, runner: Path, expected_model: str) -> dict:
    import numpy as np
    import pyarrow.parquet as parquet
    import soundfile as sf

    capture = capture.resolve(strict=True)
    corpus = corpus.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    result_path = capture / "result.json"
    prediction_path = capture / "predictions.jsonl"
    result = json.loads(result_path.read_text())
    prepared = json.loads(corpus.read_text())
    if result["completed"] is not True or prepared["passed"] is not True:
        raise ValueError("语料准备和 ASR 测量必须完成")
    if not prepared["splits"] or type(prepared["total_rows"]) is not int or prepared["total_rows"] <= 0:
        raise ValueError("ASR 语料必须包含实际录音记录")
    if any(type(item["rows"]) is not int or item["rows"] <= 0 for item in prepared["splits"].values()):
        raise ValueError("每项语料必须声明正整数的录音数量")
    if set(result["measurements"]) != set(prepared["splits"]):
        raise ValueError("ASR 汇总必须覆盖全部声明的语料")
    if not expected_model.strip():
        raise ValueError("必须声明实际 ASR 模型来源")
    if result["model_revision"] != expected_model:
        raise ValueError("ASR 模型来源与声明不一致")
    revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if result["source_revision"] != revision or result["qwen_module_sha256"] != file_sha256(
        source_root / "src/oh_my_duck/voice/qwen.py"
    ):
        raise ValueError("ASR 实现源码与测量记录不一致")
    if result["preparation_sha256"] != file_sha256(corpus) or (
        result["predictions_sha256"] != file_sha256(prediction_path) or
        result["runner_sha256"] != file_sha256(runner.resolve(strict=True))
    ):
        raise ValueError("ASR 测量文件 SHA256 不一致")
    for field in ("dataset", "dataset_revision", "split"):
        if result[field] != prepared[field]:
            raise ValueError(f"语料 {field} 与测量记录不一致")
    if prepared["dataset"] != "google/fleurs" or prepared["split"] != "test":
        raise ValueError("当前复核入口要求 FLEURS test 语料")
    if result["device"] != "cpu" or result["dtype"] != "torch.float32" or (
        result["cuda_initialized"] is not False or result["gpu_acceptance_performed"] is not False
    ):
        raise ValueError("当前复核入口要求明确的 CPU Float32 测量记录")
    records = {name: [] for name in prepared["splits"]}
    with prediction_path.open() as predictions:
        for line in predictions:
            record = json.loads(line)
            language = record["language"]
            if language not in records or type(record["row_index"]) is not int or (
                record["row_index"] != len(records[language])
            ):
                raise ValueError("语料测量编号必须连续且没有重复")
            records[language].append(record)
    measurements = {}
    audio_files = set()
    for language, definition in prepared["splits"].items():
        source_path = Path(definition["path"]).resolve(strict=True)
        if file_sha256(source_path) != definition["sha256"]:
            raise ValueError("原始 Parquet 文件 SHA256 不一致")
        table = parquet.ParquetFile(source_path)
        if table.metadata.num_rows != definition["rows"] or len(records[language]) != definition["rows"]:
            raise ValueError("ASR 测量必须覆盖声明语料中的全部记录")
        for index, batch in enumerate(table.iter_batches(batch_size=1)):
            original = batch.to_pylist()[0]
            record = records[language][index]
            for field, expected in (("source_id", original["id"]),
                                    ("source_audio_path", original["audio"]["path"]),
                                    ("reference", original["transcription"]),
                                    ("raw_reference", original["raw_transcription"]),
                                    ("source_audio_sha256", hashlib.sha256(original["audio"]["bytes"]).hexdigest())):
                if type(record[field]) is not type(expected) or record[field] != expected:
                    raise ValueError(f"ASR {field} 与原始语料不一致")
            audio_path = Path(record["audio_path"]).resolve(strict=True)
            if not audio_path.is_relative_to(capture / "audio") or audio_path in audio_files:
                raise ValueError("每条录音必须使用 capture/audio 中独立的文件")
            audio_files.add(audio_path)
            if file_sha256(audio_path) != record["audio_sha256"]:
                raise ValueError("ASR 输入音频 SHA256 不一致")
            source_audio, source_rate = sf.read(BytesIO(original["audio"]["bytes"]),
                                               dtype="float32", always_2d=True)
            audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
            info = sf.info(audio_path)
            if any(type(record[name]) is not int or record[name] <= 0
                   for name in ("sample_rate", "frames", "channels")):
                raise ValueError("ASR 音频元数据必须使用正整数")
            if info.format != "WAV" or info.subtype != "FLOAT" or not np.isfinite(audio).all() or (
                rate != source_rate or not np.array_equal(audio, source_audio)
            ):
                raise ValueError("ASR 输入必须保留原始音频的全部有限采样值")
            if record["sample_rate"] != rate or record["frames"] != len(audio) or (
                record["channels"] != audio.shape[1] or record["decoded_samples_equal"] is not True or
                record["duration_s"] != len(audio) / rate or not record["hypothesis"].strip()
            ):
                raise ValueError("ASR 音频元数据或转写文本不一致")
            for field, text in (("normalized_reference", record["reference"]),
                                ("normalized_hypothesis", record["hypothesis"])):
                if record[field] != normalize_text(text):
                    raise ValueError("ASR 文字规范化与原始文字不一致")
        metrics = calculate_metrics(records[language], english=language == "en_us")
        if metrics != result["measurements"][language]:
            raise ValueError("ASR 汇总指标与逐条测量不一致")
        measurements[language] = metrics
    if sum(len(values) for values in records.values()) != prepared["total_rows"]:
        raise ValueError("语料总数与原始定义不一致")
    return {
        "passed": True, "scope": "Full recorded FLEURS CPU ASR measurements and original audio audit",
        "source_revision": revision, "model_revision": expected_model,
        "dataset": prepared["dataset"], "dataset_revision": prepared["dataset_revision"],
        "records": prepared["total_rows"], "measurements": measurements,
        "capture_result_sha256": file_sha256(result_path),
        "predictions_sha256": file_sha256(prediction_path), "corpus_sha256": file_sha256(corpus),
        "runner_sha256": file_sha256(runner),
        "checker_sha256": file_sha256(Path(__file__)),
        "new_model_inference": False, "gpu_acceptance_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="独立复核全部真实录音、ASR 转写与识别指标")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_corpus(args.capture, args.corpus, args.source_root, args.runner, args.expected_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({name: result[name] for name in ("passed", "records", "model_revision")}))
    return 0


if __name__ == "__main__":
    main()
