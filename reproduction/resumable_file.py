"""Bounded HTTP range requests with resume and an expected whole-file checksum."""
import hashlib
import os
from pathlib import Path
import re
import time
import requests
import subprocess


def download_file(url, destination, expected_size, sha256, existing_partial=None):
    dest=Path(destination)
    dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_name(dest.name+'.part')
    if not dest.exists() and not part.exists() and existing_partial:
        existing_partial=Path(existing_partial)
        if existing_partial.exists():existing_partial.rename(part)
    aria_root=Path('/root/autodl-tmp/starvla/tools/aria2')
    aria=aria_root/'usr/bin/aria2c'
    if not dest.exists() and aria.exists():
        env=os.environ.copy()
        env['LD_LIBRARY_PATH']=str(aria_root/'usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        for key in ('http_proxy','https_proxy','all_proxy','no_proxy'):
            if key not in env and key.upper() in env:env[key]=env[key.upper()]
        subprocess.run([str(aria),'--no-conf=true','--no-netrc=true','--enable-color=false','--show-console-readout=false','--console-log-level=warn','--summary-interval=30','--download-result=full','--continue=true','--auto-file-renaming=false','--file-allocation=none','--max-connection-per-server=16','--split=16','--min-split-size=8M','--disk-cache=8M','--connect-timeout=30','--timeout=60','--max-tries=100', '--lowest-speed-limit=16K','--retry-wait=3','--check-integrity=true','--checksum=sha-256='+sha256,'--dir='+str(part.parent),'--out='+part.name,url],env=env,check=True)
    elif not aria.exists() and Path(str(part)+'.aria2').exists():
        raise RuntimeError('An aria2 control file exists; restore aria2 before resuming this sparse file')
    if not dest.exists():
        failures=0
        session=requests.Session()
        while not part.exists() or part.stat().st_size<expected_size:
            offset=part.stat().st_size if part.exists() else 0
            end=min(expected_size-1,offset+16*1024**2-1)
            try:
                with session.get(url,headers={'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(30,90)) as response:
                    response.raise_for_status()
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if response.status_code!=206 or not match:
                        raise RuntimeError(f'Expected range response, got status {response.status_code}, Content-Range={response.headers.get("Content-Range")}')
                    start,actual_end,total=map(int,match.groups())
                    if start!=offset or actual_end>end or total!=expected_size:
                        raise RuntimeError(f'Unexpected byte range {match.group()} for {offset}-{end}/{expected_size}')
                    remaining=actual_end-offset+1
                    with part.open('ab') as stream:
                        for block in response.iter_content(1024**2):
                            if len(block)>remaining:raise RuntimeError('Response exceeded declared range')
                            stream.write(block)
                            remaining-=len(block)
                    if remaining:raise OSError(f'Incomplete response: {remaining} bytes missing')
                failures=0
                print(f'RANGE {dest.name} {part.stat().st_size}/{expected_size}',flush=True)
            except Exception as exc:
                failures+=1
                print(f'RETRY {dest.name} #{failures}: {type(exc).__name__}: {exc}',flush=True)
                if failures>=20:raise
                time.sleep(min(30,failures*3))
        target=part
    else:target=dest
    if target.stat().st_size!=expected_size:
        raise ValueError(f'Size mismatch: {target}')
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=sha256:
        raise ValueError(f'SHA256 mismatch: {target}; file preserved for investigation')
    if target==part:os.replace(part,dest)
    print(f'HASH_VERIFIED {dest}',flush=True)
    return dest
