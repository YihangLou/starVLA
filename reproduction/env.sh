#!/usr/bin/env bash
export STARVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export STARVLA_STORAGE=/root/autodl-tmp/starvla
export HF_HOME="$STARVLA_STORAGE/cache/huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=60
export PIP_CACHE_DIR="$STARVLA_STORAGE/cache/pip"
mkdir -p "$STARVLA_STORAGE/tmp"
export TMPDIR="$STARVLA_STORAGE/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TORCH_HOME="$STARVLA_STORAGE/cache/torch"
export CUDA_HOME=/usr/local/cuda
export PATH="$STARVLA_ROOT/envs/starVLA/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$STARVLA_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
export LIBERO_CONFIG_PATH="$STARVLA_STORAGE/libero_config"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export XDG_CACHE_HOME="$STARVLA_STORAGE/cache"
export TORCH_EXTENSIONS_DIR="$STARVLA_STORAGE/cache/torch_extensions"
export TRITON_CACHE_DIR="$STARVLA_STORAGE/cache/triton"
export CUDA_CACHE_PATH="$STARVLA_STORAGE/cache/cuda"

# The CPU image lacks the GLVND EGL loader. Keep its Ubuntu package on the data disk.
if [[ -f "$STARVLA_STORAGE/tools/egl/usr/lib/x86_64-linux-gnu/libEGL.so.1" ]]; then
  export LD_LIBRARY_PATH="$STARVLA_STORAGE/tools/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export NO_ALBUMENTATIONS_UPDATE=1
