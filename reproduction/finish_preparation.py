"""Observe known jobs, then run CPU verification once; never restart a job."""
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parent
jobs_path=ROOT/'preparation_jobs.json'
jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}

def alive(job):
    try:
        fields=Path(f"/proc/{job['pid']}/stat").read_text().split()
        return fields[21]==job['proc_start_ticks'] and fields[2]!='Z'
    except FileNotFoundError:return False

def wait_for(names):
    last=None
    while True:
        running=[name for name in names if name in jobs and alive(jobs[name])]
        if running!=last:
            print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'waiting for',running,flush=True)
            last=running
        if not running:return
        time.sleep(30)

wait_for(['libero_download','asset_download','training_install','evaluation_install','download_recovery'])
for name in ('training_environment.lock.txt','evaluation_environment.lock.txt','dataset_verification.json','asset_verification.json'):
    assert (ROOT/name).is_file(),f'Job ended without required completion artifact: {name}. Check install/download logs.'
subprocess.run([str(ROOT/'envs/starVLA/bin/python'),str(ROOT/'verify_preparation.py')],check=True)
subprocess.run([str(ROOT/'envs/libero/bin/python'),'-m','pip','check'],check=True)
subprocess.run([str(ROOT/'envs/libero/bin/python'),str(ROOT/'verify_evaluation.py')],check=True)
print('CPU preparation checks completed. GPU forward/backward and rendering still require attached hardware.',flush=True)
