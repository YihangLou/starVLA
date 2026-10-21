#!/usr/bin/env bash
# Bound CPU-library thread pools in small preparation containers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
CPU=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
exec taskset -c "$CPU" python -u "$ROOT/run_final_preflight.py" "$@"
