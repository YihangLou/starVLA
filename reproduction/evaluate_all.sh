#!/usr/bin/env bash
# Start the model server separately with start_policy_server.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
export LIBERO_HOME="${LIBERO_HOME:-$ROOT/LIBERO}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-$ROOT/envs/libero/bin/python}"
export CKPT="${CKPT:-$ROOT/source/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-6694}"
export NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
test -f "$CKPT"
for suite in libero_spatial libero_object libero_goal libero_10; do
  TASK_SUITE_NAME="$suite" bash "$ROOT/source/examples/LIBERO/eval_files/eval_libero.sh" \
    2>&1 | tee "$ROOT/logs/07_eval_${suite}.log"
done
