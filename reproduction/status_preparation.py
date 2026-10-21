"""Report observed process state and completed verification artifacts."""
import datetime
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
jobs_path=ROOT/'preparation_jobs.json'
jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
report={'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':{},'verified':{},'training_started':False}
for name,job in jobs.items():
    if not isinstance(job,dict):continue
    try:
        fields=Path(f"/proc/{job['pid']}/stat").read_text().split()
        state='live' if fields[21]==job['proc_start_ticks'] and fields[2]!='Z' else 'ended'
    except FileNotFoundError:state='ended'
    report['jobs'][name]={'pid':job['pid'],'state':state,'log':job['log']}
for name,file in [('libero','dataset_verification.json'),('models_and_vlm','asset_verification.json')]:
    path=ROOT/file
    report['verified'][name]=[r['repo'] for r in json.loads(path.read_text())] if path.exists() else []
for name,file in [('training_environment','training_environment.lock.txt'),('evaluation_environment','evaluation_environment.lock.txt'),('cpu_preflight','preparation_verification.json'),('evaluation_preflight','evaluation_verification.json')]:
    report['verified'][name]=(ROOT/file).exists()
readiness=ROOT/'preparation_readiness.json'
if readiness.exists():
    final=json.loads(readiness.read_text())
    report['ready_for_gpu_restart']=final['ready_for_gpu_restart']
    report['incomplete_cpu_checks']=list(final['incomplete_cpu_checks'])
report['free_gib']={p:round(shutil.disk_usage(p).free/1024**3,2) for p in ('/','/root/autodl-tmp')}
(ROOT/'preparation_progress.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
