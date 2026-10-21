#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
python -c 'import tempfile; assert tempfile.gettempdir().startswith("/root/autodl-tmp/")'
# CUDA 12.8 builds support the attached Blackwell sm_120 GPU.
python -m pip install 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128
LOCAL_WHEELS=()
for stem in pyarrow-14.0.1 av-12.3.0 opencv_python_headless-4.11.0.86 numpy-1.26.4; do
  matches=("$STARVLA_STORAGE/cache/wheels/$stem-"*.whl)
  if [[ -f "${matches[0]}" ]]; then LOCAL_WHEELS+=("${matches[0]}"); fi
done
python -m pip install -r "$ROOT/requirements.training.txt" -c "$ROOT/constraints.txt" "${LOCAL_WHEELS[@]}" --index-url https://pypi.org/simple
# The upstream decord wheel and pipablepytorch3d wheel have incompatible tags.
# eva-decord provides the same decord module; remove its overlapping predecessor.
if python -c 'from importlib.metadata import version; version("decord")' 2>/dev/null; then
  python -m pip uninstall -y decord eva-decord
  python -m pip install eva-decord==0.6.1 --index-url https://pypi.org/simple
fi
python -m pip uninstall -y pipablepytorch3d
PYTORCH3D_SOURCE="$STARVLA_STORAGE/tools/pytorch3d-0.7.6"
if [[ ! -f "$PYTORCH3D_SOURCE/setup.py" ]]; then
  PYTORCH3D_SOURCE='https://github.com/facebookresearch/pytorch3d/archive/f34104cf6ebefacd7b7e07955ee7aaa823e616ac.tar.gz'
fi
PYTORCH3D_NO_EXTENSION=1 python -m pip install "$PYTORCH3D_SOURCE" --no-deps --no-build-isolation
python -m pip install ninja packaging --index-url https://pypi.org/simple
FLASH_WHEEL="$STARVLA_STORAGE/cache/wheels/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
if [[ -f "$FLASH_WHEEL" ]]; then
  python -m pip install "$FLASH_WHEEL"
else
  python -m pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
fi
python -m pip install -e "$ROOT/source" --no-deps --no-build-isolation
python -m pip check
python -m pip freeze > "$ROOT/training_environment.lock.txt"
