#!/usr/bin/env bash
# Attach an existing prepared data disk without copying models or environments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(dirname "$ROOT")"
STORAGE="${STARVLA_STORAGE:-/root/autodl-tmp/starvla}"
[[ -d "$STORAGE" ]] || { echo "Prepared storage is missing: $STORAGE" >&2; exit 1; }

ensure_link() {
  local link="$1" target="$2"
  if [[ -L "$link" ]]; then
    [[ "$(readlink -m "$link")" == "$(readlink -m "$target")" ]] || {
      echo "Existing symlink points elsewhere: $link" >&2; return 1;
    }
  elif [[ -e "$link" ]]; then
    echo "Refusing to replace an existing path: $link" >&2
    return 1
  else
    ln -s "$target" "$link"
  fi
}

for name in envs conda_pkgs data LIBERO results; do
  ensure_link "$ROOT/$name" "$STORAGE/$name"
done
if [[ ! -e "$ROOT/source" && ! -L "$ROOT/source" ]]; then
  ln -s .. "$ROOT/source"
fi
[[ "$(readlink -m "$ROOT/source")" == "$REPO" ]]
mkdir -p "$ROOT/logs" "$REPO/playground/Datasets"
ensure_link "$REPO/playground/Pretrained_models" "$STORAGE/models"
ensure_link "$REPO/playground/Datasets/LEROBOT_LIBERO_DATA" "$STORAGE/data/libero"
ensure_link "$REPO/playground/Datasets/LLaVA-OneVision-COCO" "$STORAGE/data/LLaVA-OneVision-COCO"

# Existing Conda entry points embed this original prefix in their shebangs.
if [[ "${1:-}" == --legacy-alias ]]; then
  LEGACY=/root/workspace/flow_matching/starvla_reproduction
  mkdir -p "$(dirname "$LEGACY")"
  ensure_link "$LEGACY" "$ROOT"
fi
echo "Local storage links ready: $ROOT"
