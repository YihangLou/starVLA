"""Download a pinned PyPI wheel with Ubuntu-packaged aria2 and verify SHA256."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
ROOT=Path('/root/autodl-tmp/starvla')
name,version=sys.argv[1:3]
r=requests.get(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60);r.raise_for_status()
tag_order={tag: i for i, tag in enumerate(sys_tags())}
supported=set(tag_order)
files=[f for f in r.json()['urls'] if f['filename'].endswith('.whl') and parse_wheel_filename(f['filename'])[3] & supported]
assert files, 'No wheel supports this interpreter'
files.sort(key=lambda item: min(tag_order[t] for t in parse_wheel_filename(item['filename'])[3] if t in supported))
f=files[0]
directory=ROOT/'cache/wheels';directory.mkdir(parents=True,exist_ok=True)
path=directory/f['filename']
if not path.exists():
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']=str(ROOT/'tools/aria2/usr/lib/x86_64-linux-gnu')+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    for key in ('http_proxy','https_proxy','all_proxy','no_proxy'):
        if key not in env and key.upper() in env:env[key]=env[key.upper()]
    start=time.monotonic()
    subprocess.run([str(ROOT/'tools/aria2/usr/bin/aria2c'),'--no-conf=true','--no-netrc=true','--enable-color=false','--console-log-level=warn','--summary-interval=15','--download-result=full','--continue=true','--auto-file-renaming=false','--file-allocation=none','--max-connection-per-server=16','--split=16','--min-split-size=4M','--disk-cache=8M','--connect-timeout=30','--timeout=60','--max-tries=100', '--lowest-speed-limit=16K','--retry-wait=3','--check-integrity=true','--checksum=sha-256='+f['digests']['sha256'],'--dir='+str(directory),'--out='+f['filename']+'.download',f['url']],env=env,check=True)
    partial=path.with_name(path.name+'.download')
    assert partial.stat().st_size==f['size']
    partial.rename(path)
    print('elapsed_seconds',round(time.monotonic()-start,2),flush=True)
hash_value=hashlib.sha256()
with path.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024**2),b''):hash_value.update(block)
assert hash_value.hexdigest()==f['digests']['sha256']
print('VERIFIED',path,flush=True)
