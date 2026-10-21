#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
export STARVLA_PYTHON="${STARVLA_PYTHON:-$ROOT/envs/starVLA/bin/python}"
export CKPT="${CKPT:-$ROOT/source/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}"
test -f "$CKPT"
bash "$ROOT/source/examples/LIBERO/eval_files/run_policy_server.sh" \
  2>&1 | tee "$ROOT/logs/07_policy_server.log"
