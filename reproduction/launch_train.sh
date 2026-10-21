#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
source "$ROOT/env.sh"
PYTHON="${STARVLA_PYTHON:-$ROOT/envs/starVLA/bin/python}"
FRAMEWORK="${1:-QwenOFT}"
MODE="${2:-vla}"
case "$FRAMEWORK" in QwenOFT|QwenFast|QwenPI|QwenGR00T) ;; *) echo "Use QwenOFT, QwenFast, QwenPI, or QwenGR00T"; exit 2;; esac
case "$MODE" in vla) TRAINER=train_starvla.py;; cotrain) TRAINER=train_starvla_cotrain.py;; *) exit 2;; esac
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):$PATH"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="$ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT/source"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "No usable CUDA GPU: training has not started"'
NUM_PROCESSES="${NUM_PROCESSES:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
RUN_ID="${RUN_ID:-${FRAMEWORK}_${MODE}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT="$ROOT/results/Checkpoints/$RUN_ID"
mkdir -p "$OUTPUT"
cp "$0" "$OUTPUT/"
"$PYTHON" - "$ROOT" "$NUM_PROCESSES" "$OUTPUT" <<'PY'
import sys, yaml
from pathlib import Path
root, count, output = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
import torch
assert 0 < count <= torch.cuda.device_count()
cfg = yaml.safe_load((root / "source/starVLA/config/deepseeds/deepspeed_zero2.yaml").read_text())
cfg["num_processes"] = count
(output / "accelerate.yaml").write_text(yaml.safe_dump(cfg))
PY
EXTRA=()
if [[ "${SMOKE_ONLY:-0}" == 1 ]]; then
  EXTRA+=(--trainer.max_train_steps 2 --trainer.num_warmup_steps 0 --trainer.save_interval 1 --trainer.eval_interval 1 --trainer.logging_frequency 1)
fi
"$PYTHON" -m accelerate.commands.launch \
  --config_file "$OUTPUT/accelerate.yaml" --num_processes "$NUM_PROCESSES" \
  "starVLA/training/$TRAINER" \
  --config_yaml "$ROOT/configs/${FRAMEWORK}_${MODE}.yaml" \
  --run_id "$RUN_ID" "${EXTRA[@]}" 2>&1 | tee "$OUTPUT/launch.log"
