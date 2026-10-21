"""Finish training installation after the verified CUDA wheel downloads stop."""
from pathlib import Path
import json
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
STORAGE=Path('/root/autodl-tmp/starvla')
WATCH_PID=int(sys.argv[1])
WATCH_START=sys.argv[2]
while True:
    try:
        fields=Path(f'/proc/{WATCH_PID}/stat').read_text().split()
        live=fields[21]==WATCH_START and fields[2]!='Z'
    except FileNotFoundError:live=False
    if not live:break
    time.sleep(5)
wheel=STORAGE/'cache/wheels/torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl'
assert wheel.is_file(), 'CUDA torch wheel is incomplete'
# This exact path prevents selecting the CPU build kept in the same wheelhouse.
subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links',str(wheel.parent),str(wheel),'torchvision==0.21.0'],check=True)
subprocess.run(['bash',str(ROOT/'install_training_env.sh')],check=True)
print('Training environment installation finished',flush=True)
