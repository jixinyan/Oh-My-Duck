import argparse
import hashlib
import json
import posixpath
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from pxr import UsdUtils


def read_remote(url, etag=None):
    headers = {"If-Match": f'"{etag}"'} if etag else {}
    response = requests.get(url, headers=headers, timeout=(30, 180))
    response.raise_for_status()
    received_etag = response.headers.get("ETag", "").strip('"')
    if etag and received_etag != etag:
        raise ValueError(f"Asset ETag changed: {url}")
    return response.content, received_etag, response.headers.get("Last-Modified")


def list_objects(origin, prefix):
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    token = None
    objects = []
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        response = requests.get(origin, params=query, timeout=(30, 180))
        response.raise_for_status()
        page = ElementTree.fromstring(response.content)
        for item in page.findall("s3:Contents", namespace):
            objects.append({
                "key": item.findtext("s3:Key", namespaces=namespace),
                "size": int(item.findtext("s3:Size", namespaces=namespace)),
                "etag": item.findtext("s3:ETag", namespaces=namespace).strip('"'),
                "last_modified": item.findtext("s3:LastModified", namespaces=namespace),
            })
        if page.findtext("s3:IsTruncated", namespaces=namespace) != "true":
            break
        token = page.findtext("s3:NextContinuationToken", namespaces=namespace)
    if not objects:
        raise FileNotFoundError(f"Empty asset directory: {origin}/{prefix}")
    return objects


def verify_content(content, item, previous):
    sha256 = hashlib.sha256(content).hexdigest()
    if len(content) != item["size"]:
        raise ValueError(f"Asset size differs from inventory: {item['key']}")
    if previous is not None:
        if sha256 != previous["sha256"]:
            raise ValueError(f"Cached asset SHA256 changed: {item['key']}")
    elif len(item["etag"]) == 32:
        if hashlib.md5(content, usedforsecurity=False).hexdigest() != item["etag"]:
            raise ValueError(f"Asset content differs from inventory ETag: {item['key']}")
    else:
        raise ValueError(f"Cached asset requires a previous SHA256 manifest: {item['key']}")
    return sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/simulation-scenes.json"))
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    catalog = json.loads(args.catalog.read_text())
    scene = catalog["scenes"][args.scene]
    destination = args.destination or Path(catalog["asset_root"]) / args.scene
    source_root = destination / "source"
    origin = scene["asset_origin"]
    objects = list_objects(origin, scene["asset_prefix"])
    destination.mkdir(parents=True, exist_ok=True)
    inventory = {"scene_id": args.scene, "source": scene, "objects": objects}
    inventory_path = destination / "inventory.json"
    if inventory_path.exists():
        existing = json.loads(inventory_path.read_text())
        if existing["objects"] != objects:
            raise ValueError("Remote asset inventory changed")
    else:
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"scene_id": args.scene, "files": len(objects), "bytes": sum(item["size"] for item in objects)}), flush=True)
    if args.inventory_only:
        return
    manifest_path = destination / "asset-provenance.json"
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    previous_files = {item["key"]: item for item in previous_manifest["files"]} if previous_manifest else {}

    def acquire(item):
        local_path = source_root / item["key"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            content = local_path.read_bytes()
            previous = previous_files.get(item["key"])
            if previous is None and len(item["etag"]) != 32:
                remote_content, _, _ = read_remote(f"{origin}/{item['key']}", item["etag"])
                if content != remote_content:
                    raise ValueError(f"Cached asset differs from remote content: {item['key']}")
                sha256 = hashlib.sha256(content).hexdigest()
            else:
                sha256 = verify_content(content, item, previous)
        else:
            content, _, _ = read_remote(f"{origin}/{item['key']}", item["etag"])
            if len(content) != item["size"]:
                raise ValueError(f"Downloaded asset size differs: {item['key']}")
            sha256 = hashlib.sha256(content).hexdigest()
            local_path.write_bytes(content)
        return {**item, "sha256": sha256}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        files = list(executor.map(acquire, objects))
    file_keys = {item["key"] for item in files}
    root_path = source_root / scene["usd_relative_path"]
    if not root_path.is_file():
        raise FileNotFoundError(root_path)
    runtime_dependencies = set()
    external_files = {}

    def inspect_dependency(layer, dependency):
        asset_path = dependency.assetPath
        if asset_path in catalog["runtime_mdl_modules"]:
            runtime_dependencies.add(asset_path)
            return dependency
        relative_layer = Path(layer.realPath).relative_to(source_root).as_posix()
        asset_url = urljoin(f"{origin}/{relative_layer}", asset_path)
        parsed = urlparse(asset_url)
        if parsed.scheme != "https" or parsed.netloc != urlparse(origin).netloc:
            raise ValueError(f"Unsupported external asset origin: {asset_url}")
        key = posixpath.normpath(parsed.path).lstrip("/")
        if key in file_keys or key in external_files:
            return dependency
        local_path = source_root / key
        previous = previous_files.get(key)
        if local_path.exists() and previous is not None:
            verify_content(local_path.read_bytes(), previous, previous)
            external_files[key] = previous
        else:
            response = requests.head(asset_url, timeout=(30, 180))
            response.raise_for_status()
            etag = response.headers["ETag"].strip('"')
            content, _, modified = read_remote(asset_url, etag)
            if local_path.exists() and local_path.read_bytes() != content:
                raise ValueError(f"Cached external asset changed: {key}")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                local_path.write_bytes(content)
            external_files[key] = {"key": key, "size": len(content), "sha256": hashlib.sha256(content).hexdigest(), "etag": etag, "last_modified": modified, "source_url": asset_url}
        return dependency

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root_path), inspect_dependency)
    unresolved_assets = set(unresolved) - set(catalog["runtime_mdl_modules"])
    if unresolved_assets:
        raise FileNotFoundError(f"Unresolved USD dependencies: {sorted(unresolved_assets)}")
    manifest = {
        "schema_version": 1,
        "scene_id": args.scene,
        "source": scene,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "usd_path": str(root_path.resolve()),
        "files": sorted(files + list(external_files.values()), key=lambda item: item["key"]),
        "usd_layers_checked": len(layers),
        "usd_assets_checked": len(assets),
        "runtime_mdl_modules": sorted(runtime_dependencies),
    }
    if previous_manifest:
        if previous_manifest["files"] != manifest["files"]:
            raise ValueError("Acquired asset content changed")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"scene_id": args.scene, "usd_path": str(root_path.resolve()), "provenance": str(manifest_path.resolve()), "completed_files": len(manifest["files"])}), flush=True)


if __name__ == "__main__":
    main()
