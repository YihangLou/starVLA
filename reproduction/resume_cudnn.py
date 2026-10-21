"""Resume the preserved CUDA wheel on the data disk, using PyPI SHA256 metadata."""
from pathlib import Path
import hashlib
import os
import time
import requests

ROOT=Path('/root/autodl-tmp/starvla')
name='nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl'
dest=ROOT/'cache/wheels'/name
dest.parent.mkdir(parents=True,exist_ok=True)
part=dest.with_suffix('.whl.part')
preserved=ROOT/'tmp/interrupted_install/pip-unpack-rcgmt_2m'/name
if not part.exists() and preserved.exists():preserved.rename(part)
response=requests.get('https://pypi.org/pypi/nvidia-cudnn-cu12/9.1.0.70/json',timeout=60)
response.raise_for_status()
info=next(f for f in response.json()['urls'] if f['filename']==name)
size=info['size']
if not dest.exists():
    failures=0
    while not part.exists() or part.stat().st_size<size:
        offset=part.stat().st_size if part.exists() else 0
        end=min(size-1,offset+16*1024**2-1)
        try:
            with requests.get(info['url'],headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(30,120)) as response:
                response.raise_for_status()
                assert response.status_code==206,response.status_code
                assert response.headers['Content-Range'].startswith(f'bytes {offset}-'),response.headers.get('Content-Range')
                remaining=end-offset+1
                with part.open('ab') as out:
                    for block in response.iter_content(1024**2):
                        assert len(block)<=remaining
                        out.write(block);remaining-=len(block)
                if remaining:raise OSError(f'Partial response: {remaining} bytes missing')
            failures=0
            print(f'{name}: {part.stat().st_size}/{size} bytes',flush=True)
        except Exception as exc:
            failures+=1
            print('Retry',failures,repr(exc),flush=True)
            if failures>=20:raise
            time.sleep(min(30,failures*3))
    assert part.stat().st_size==size
    target=part
else:target=dest
h=hashlib.sha256()
with target.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
assert h.hexdigest()==info['digests']['sha256'],'CUDA wheel SHA256 mismatch'
if target==part:part.rename(dest)
print('Verified',dest,flush=True)
