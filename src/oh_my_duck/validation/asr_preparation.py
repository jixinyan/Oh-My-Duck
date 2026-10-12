import argparse
import json
from pathlib import Path
import re

from oh_my_duck.validation.asr_corpus import file_sha256


def revision_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("revision 必须是完整的 Git SHA")
    return value


def prepare_corpus(cache: Path, output: Path, revision: str, languages: list[str]) -> dict:
    if not languages or len(languages) != len(set(languages)):
        raise ValueError("语料语言必须有内容且没有重复")
    if output.exists():
        raise FileExistsError(output)
    from huggingface_hub import DatasetCard, HfApi, snapshot_download
    import pyarrow.parquet as parquet

    info = HfApi().dataset_info("google/fleurs", revision=revision, files_metadata=True)
    if info.sha != revision or info.card_data["license"] != ["cc-by-4.0"]:
        raise ValueError("语料 revision 或许可与声明不一致")
    files = ["README.md", *(f"parquet-data/{name}/test-00000-of-00001.parquet" for name in languages)]
    metadata = {item.rfilename: item for item in info.siblings}
    if any(name not in metadata for name in files):
        raise ValueError("固定语料来源缺少所选语言的完整 test 文件")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = Path(snapshot_download(
        "google/fleurs", repo_type="dataset", revision=revision, allow_patterns=files,
        cache_dir=str(cache.expanduser().resolve()), max_workers=2,
    )).resolve(strict=True)
    if snapshot.name != revision:
        raise ValueError("下载的语料 snapshot 与固定 revision 不一致")
    card = DatasetCard.load(str(snapshot / "README.md")).data.to_dict()
    if card["license"] != ["cc-by-4.0"]:
        raise ValueError("语料文件声明的许可不一致")
    configs = {item["config_name"]: item for item in card["dataset_info"]}
    splits = {}
    for name in languages:
        path = snapshot / f"parquet-data/{name}/test-00000-of-00001.parquet"
        expected = next(item["num_examples"] for item in configs[name]["splits"] if item["name"] == "test")
        table = parquet.ParquetFile(path)
        upstream = metadata[path.relative_to(snapshot).as_posix()]
        digest = file_sha256(path)
        if type(expected) is not int or expected <= 0 or table.metadata.num_rows != expected:
            raise ValueError("原始语料数量与来源声明不一致")
        if upstream.lfs is None or digest != upstream.lfs.sha256 or path.stat().st_size != upstream.size:
            raise ValueError("原始语料文件尺寸或 SHA256 与上游不一致")
        first = next(table.iter_batches(batch_size=1)).to_pylist()[0]
        if not isinstance(first["audio"]["bytes"], bytes) or not first["audio"]["bytes"]:
            raise ValueError("原始语料必须包含实际录音")
        splits[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size,
                        "rows": table.metadata.num_rows, "columns": table.schema_arrow.names}
    result = {"passed": True, "dataset": "google/fleurs", "dataset_revision": revision,
              "license": card["license"], "split": "test", "snapshot": str(snapshot),
              "readme_sha256": file_sha256(snapshot / "README.md"), "splits": splits,
              "total_rows": sum(item["rows"] for item in splits.values()),
              "asr_inference_performed": False, "gpu_acceptance_performed": False}
    with (output / "result.json").open("x") as file:
        json.dump(result, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="下载固定 FLEURS test 语料并核验上游文件与完整数量")
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", type=revision_argument, required=True)
    parser.add_argument("--languages", nargs="+", default=["cmn_hans_cn", "en_us"])
    args = parser.parse_args()
    result = prepare_corpus(args.cache_directory, args.output.resolve(), args.dataset_revision, args.languages)
    print(json.dumps({name: result[name] for name in ("passed", "dataset_revision", "total_rows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
