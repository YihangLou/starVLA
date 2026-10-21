"""Let existing download jobs finish; resume only failed/incomplete snapshots."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
jobs_path=ROOT/'preparation_jobs.json'
jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
def finish(name,script,report,count):
    job=jobs.get(name)
    while job:
        try:
            fields=Path(f"/proc/{job['pid']}/stat").read_text().split()
            if fields[21]!=job['proc_start_ticks'] or fields[2]=='Z':break
        except FileNotFoundError:break
        time.sleep(15)
    path=ROOT/report
    if path.exists() and len(json.loads(path.read_text()))==count:
        print(name,'already complete',flush=True)
        return
    print(name,'previous process ended; resuming partial downloads',flush=True)
    with (ROOT/'logs'/('16_resume_'+name+'.log')).open('a') as log:
        subprocess.run([sys.executable,str(ROOT/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(finish,'libero_download','download_libero.py','dataset_verification.json',4),pool.submit(finish,'asset_download','download_assets.py','asset_verification.json',4)]
    for future in concurrent.futures.as_completed(futures):future.result()
