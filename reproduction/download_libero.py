"""Download the four pinned, complete LIBERO snapshots and verify every file."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "download_manifest.json").read_text())
reports = []
for repo in manifest:
    if repo["type"] != "dataset" or not repo["repo"].startswith("IPEC-COMMUNITY/"):
        continue
    dest = ROOT / "data/libero" / repo["repo"].split("/")[-1]
    remaining = sum(
        f["size"] for f in repo["files"]
        if not (dest / f["rfilename"]).exists()
    )
    if shutil.disk_usage(ROOT).free < remaining + 1024**3:
        raise RuntimeError(f"Insufficient disk for {repo['repo']}, reserving 1 GiB")
    for attempt in range(1, 21):
        try:
            snapshot_download(
                repo["repo"], repo_type="dataset", revision=repo["revision"],
                local_dir=dest, max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "24")),
            )
            break
        except Exception as exc:
            print("DOWNLOAD_ERROR", repo["repo"], "attempt", attempt, repr(exc), flush=True)
            if attempt == 20:
                raise
            time.sleep(min(30, attempt * 3))

    def verify(item):
        path = dest / item["rfilename"]
        assert path.stat().st_size == item["size"], str(path)
        if item.get("lfs", {}).get("sha256"):
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == item["lfs"]["sha256"], str(path)
        else:
            content = path.read_bytes()
            digest = hashlib.sha1(
                f"blob {len(content)}\0".encode() + content
            ).hexdigest()
            assert digest == item["blobId"], str(path)
        return item["size"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        total = sum(pool.map(verify, repo["files"]))
    shutil.copy2(
        ROOT / "source/examples/LIBERO/train_files/modality.json",
        dest / "meta/modality.json",
    )
    reports.append({
        "repo": repo["repo"], "revision": repo["revision"],
        "files_verified": len(repo["files"]), "bytes_verified": total,
        "hashes_verified": True, "modality_copied": True,
    })
    (ROOT / "dataset_verification.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports[-1]), flush=True)

link = ROOT / "source/playground/Datasets/LEROBOT_LIBERO_DATA"
link.parent.mkdir(parents=True, exist_ok=True)
if not os.path.lexists(link):
    link.symlink_to(ROOT / "data/libero", target_is_directory=True)
assert link.resolve() == (ROOT / "data/libero").resolve()
