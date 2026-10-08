import base64
import hashlib
from importlib.metadata import distributions
from pathlib import Path


def verify_execution_runtime(backend):
    if backend == "isaac-newton":
        return verify_usd_runtime()
    return None


def verify_usd_runtime():
    owners = []
    for installed in distributions():
        files = installed.files or []
        if any(path.parts[:1] == ("pxr",) for path in files):
            owners.append(installed)
    identities = {installed.metadata["Name"]: installed.version for installed in owners}
    if len(owners) != 1 or identities != {"usd-exchange": "3.0.0"}:
        raise RuntimeError(f"Isaac/Newton 的 OpenUSD 需要唯一的 usd-exchange 3.0.0 提供 pxr: {identities}")
    installed = owners[0]
    verified = []
    for path in installed.files:
        if path.hash is None:
            if path.parts[:1] == ("pxr",) and path.suffix != ".pyc":
                raise RuntimeError(f"OpenUSD 安装记录缺少文件 hash: {path}")
            continue
        actual = Path(installed.locate_file(path)).resolve(strict=True)
        with actual.open("rb") as source:
            digest = hashlib.file_digest(source, path.hash.mode).digest()
        if base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii") != path.hash.value:
            raise RuntimeError(f"OpenUSD 文件内容与安装记录不一致: {path}")
        verified.append(str(path))
    if not verified:
        raise RuntimeError("OpenUSD 安装记录没有可验证的文件")
    return {"provider": "usd-exchange", "version": installed.version, "files_verified": len(verified)}
