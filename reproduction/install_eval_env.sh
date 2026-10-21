#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
PYTHON="$ROOT/envs/libero/bin/python"
"$PYTHON" -m pip install 'numpy==1.24.4' 'mujoco==3.2.3' 'robosuite==1.4.0' 'bddl==1.0.1' 'gym==0.25.2' 'opencv-python==4.6.0.66' 'numba==0.60.0' 'tyro==0.9.35' 'matplotlib==3.7.5' future tqdm 'h5py==3.11.0' mediapy websockets msgpack imageio imageio-ffmpeg easydict pyyaml cloudpickle termcolor --index-url https://pypi.org/simple
CPU_WHEEL="$STARVLA_STORAGE/cache/wheels/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl"
if [[ -f "$CPU_WHEEL" ]]; then
  "$PYTHON" -m pip install "$CPU_WHEEL" --index-url https://pypi.org/simple
else
  "$PYTHON" -m pip install 'torch==2.6.0+cpu' --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
fi
"$PYTHON" -m pip install -e "$ROOT/LIBERO" --no-deps --no-build-isolation --config-settings editable_mode=compat
"$PYTHON" "$ROOT/apply_libero_compat.py"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$ROOT/evaluation_environment.lock.txt"
