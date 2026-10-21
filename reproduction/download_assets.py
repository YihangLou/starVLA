"""Fetch pinned model/VLM snapshots, hash-check files, and unpack co-training data."""
import concurrent.futures
import hashlib
import json
import os
import shutil
import zipfile
import time
import threading
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_url
from resumable_file import download_file
SNAPSHOT_LOCK = threading.Lock()
ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / 'download_manifest.json').read_text())

def fetch(repo):
    name = repo['repo'].split('/')[-1]
    dest = ROOT / ('data' if repo['type']=='dataset' else 'source/playground/Pretrained_models') / name
    dest.mkdir(parents=True, exist_ok=True)
    remaining = sum(f['size'] for f in repo['files'] if not (dest/f['rfilename']).exists())
    if shutil.disk_usage(dest).free < remaining + 10*1024**3:
        raise RuntimeError('Insufficient data-disk space')
    print('START',repo['repo'],flush=True)
    large=[f for f in repo['files'] if f['size']>32*1024**2]
    small=[f['rfilename'] for f in repo['files'] if f['size']<=32*1024**2]
    for attempt in range(1, 21):
        try:
            with SNAPSHOT_LOCK:
                snapshot_download(repo['repo'], repo_type=repo['type'], revision=repo['revision'], local_dir=dest, max_workers=4, allow_patterns=small)
            break
        except Exception as exc:
            print('DOWNLOAD_ERROR', repo['repo'], 'attempt', attempt, repr(exc), flush=True)
            if attempt == 20:
                raise
            time.sleep(min(30, attempt * 3))
    def fetch_large(item):
        sha=item['lfs']['sha256']
        partials=list((dest/'.cache/huggingface/download').rglob(f'*.{sha}.incomplete'))
        if len(partials)>1:raise RuntimeError('Multiple partial files for the same checksum')
        return download_file(hf_hub_url(repo['repo'],item['rfilename'],repo_type=repo['type'],revision=repo['revision']),dest/item['rfilename'],item['size'],sha,partials[0] if partials else None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch_large,item) for item in large]
        for future in concurrent.futures.as_completed(futures):future.result()
    for f in repo['files']:
        path=dest/f['rfilename']
        assert path.stat().st_size == f['size'],path
        sha=f.get('lfs',{}).get('sha256')
        h=hashlib.sha256() if sha else hashlib.sha1(f"blob {f['size']}\0".encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024**2),b''): h.update(block)
        assert h.hexdigest()==(sha or f['blobId']),path
    if repo['type']=='dataset':
        with zipfile.ZipFile(dest/'sharegpt4v_coco.zip') as z:
            for item in z.infolist():
                target=(dest/item.filename).resolve()
                assert target.is_relative_to(dest.resolve()),item.filename
            z.extractall(dest)
        link=ROOT/'source/playground/Datasets/LLaVA-OneVision-COCO'
        link.parent.mkdir(parents=True,exist_ok=True)
        if not os.path.lexists(link): link.symlink_to(dest,target_is_directory=True)
        assert link.resolve()==dest.resolve()
    report=dict(repo=repo['repo'],revision=repo['revision'],files_verified=len(repo['files']),bytes_verified=sum(f['size'] for f in repo['files']),hashes_verified=True,path=str(dest.resolve()))
    (ROOT/'logs'/f'{name}_verified.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFIED',repo['repo'],flush=True)
    return report

if __name__=='__main__':
    repos=[r for r in manifest if not r['repo'].startswith('IPEC-COMMUNITY/')]
    repos.sort(key=lambda r: r['repo']=='StarVLA/Qwen3-VL-OFT-LIBERO-4in1')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        reports=list(pool.map(fetch,repos))
    subprocess.run([sys.executable,str(ROOT/'prepare_vlm_annotations.py')],check=True)
    (ROOT/'asset_verification.json').write_text(json.dumps(reports,indent=2)+'\n')
