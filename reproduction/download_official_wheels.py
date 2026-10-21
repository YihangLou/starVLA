"""Resume official non-PyPI wheels; validate published hashes or wheel records."""
import base64
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile
ROOT = Path(__file__).resolve().parent
STORAGE = Path('/root/autodl-tmp/starvla')

def fetch(name):
    meta = json.loads((ROOT / 'logs' / f'{name}_asset.json').read_text())
    target = STORAGE / 'cache/wheels' / meta['filename']
    part = target.with_name(target.name + '.download')
    if not target.exists():
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(STORAGE / 'tools/aria2/usr/lib/x86_64-linux-gnu')
        args = [str(STORAGE / 'tools/aria2/usr/bin/aria2c'), '--no-conf=true', '--no-netrc=true',
                '--enable-color=false', '--show-console-readout=false', '--console-log-level=warn',
                '--summary-interval=30', '--continue=true', '--auto-file-renaming=false',
                '--file-allocation=none', '--max-connection-per-server=16', '--split=16',
                '--min-split-size=4M', '--connect-timeout=30', '--timeout=60', '--max-tries=100', '--lowest-speed-limit=16K',
                '--retry-wait=3', '--dir=' + str(part.parent), '--out=' + part.name]
        if meta['sha256']:
            args += ['--check-integrity=true', '--checksum=sha-256=' + meta['sha256']]
        subprocess.run(args + [meta['url']], env=env, check=True)
        candidate = part
    else:
        candidate = target
    assert candidate.stat().st_size == meta['size']
    digest = hashlib.file_digest(candidate.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:
        h = hashlib.sha256()
        with candidate.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
        digest = h.hexdigest()
    if meta['sha256']:
        assert digest == meta['sha256']
    with zipfile.ZipFile(candidate) as z:
        assert z.testzip() is None
        record = next(n for n in z.namelist() if n.endswith('.dist-info/RECORD'))
        for filename, checksum, size in csv.reader(io.StringIO(z.read(record).decode())):
            if checksum:
                algorithm, expected = checksum.split('=', 1)
                h = hashlib.new(algorithm)
                with z.open(filename) as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024**2), b''): h.update(chunk)
                assert base64.urlsafe_b64encode(h.digest()).decode().rstrip('=') == expected, filename
    if candidate == part: part.rename(target)
    report = dict(filename=meta['filename'], bytes=meta['size'], sha256=digest,
                  published_sha256_verified=bool(meta['sha256']), wheel_crc_and_records_verified=True)
    (ROOT/'logs'/f'{name}_wheel_verified.json').write_text(json.dumps(report, indent=2)+'\n')
    print('VERIFIED', json.dumps(report), flush=True)

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, ['cpu_torch', 'flash']))
