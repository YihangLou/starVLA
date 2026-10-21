"""Run independent CPU checks serially after downloads, recording exit codes and peak RSS."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
jobs_path=ROOT/'preparation_jobs.json'
jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
job=jobs.get('asset_download')
while job:
    try:
        fields=Path(f"/proc/{job['pid']}/stat").read_text().split()
        live=fields[21]==job['proc_start_ticks'] and fields[2]!='Z'
    except FileNotFoundError:live=False
    if not live:break
    time.sleep(2)
assert (ROOT/'asset_verification.json').exists(), 'Asset downloads did not pass final verification'
assert (ROOT/'training_environment.lock.txt').exists()
subprocess.run([sys.executable,'-m','pip','check'],check=True)
subprocess.run([sys.executable,str(ROOT/'verify_configs.py')],check=True)
required=['imports','trainer:vla','trainer:cotrain','models']+['robot:'+r['repo'].split('/')[-1] for r in json.loads((ROOT/'dataset_verification.json').read_text())]+['vlm']
parser=argparse.ArgumentParser()
parser.add_argument('--phase',action='append',choices=required,help='Rerun selected checks and retain recorded results for other checks.')
args=parser.parse_args()
phases=args.phase or required
attempts=ROOT/'preflight_attempts.json'
if attempts.exists():
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (ROOT/'logs'/f'preflight_attempts_before_{stamp}.json').write_bytes(attempts.read_bytes())
runs=json.loads(attempts.read_text()) if args.phase and attempts.exists() else {}
reports={}
completed=ROOT/'preparation_verification.json'
if completed.exists():completed.unlink()
for phase in phases:
    print('START_PHASE',phase,flush=True)
    started=time.monotonic()
    pid=os.fork()
    if pid==0:
        os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'verify_preparation.py'),'--phase',phase])
    _,status,usage=os.wait4(pid,0)
    code=os.waitstatus_to_exitcode(status)
    runs[phase]={'returncode':code,'peak_rss_mib':round(usage.ru_maxrss/1024,2),'elapsed_seconds':round(time.monotonic()-started,2),'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_affinity':sorted(os.sched_getaffinity(0))}
    print('END_PHASE',phase,json.dumps(runs[phase]),flush=True)
    if code==0:
        reports[phase]=json.loads((ROOT/'logs'/('preflight_'+phase.replace(':','_')+'.json')).read_text())
    (ROOT/'preflight_attempts.json').write_text(json.dumps(runs,indent=2)+'\n')
if all(runs.get(phase,{}).get('returncode')==0 for phase in required):
    reports={phase:json.loads((ROOT/'logs'/('preflight_'+phase.replace(':','_')+'.json')).read_text()) for phase in required}
    reports['gpu_forward_backward']='not exercised by this CPU preflight; see gpu_upgrade_verification.json for prior GPU checks'
    reports['training_started']=False
    (ROOT/'preparation_verification.json').write_text(json.dumps(reports,indent=2)+'\n')
    print('ALL_CPU_PREFLIGHT_CHECKS_PASSED',flush=True)
else:
    print('CPU_PREFLIGHT_INCOMPLETE',json.dumps(runs),flush=True)
    sys.exit(1)
