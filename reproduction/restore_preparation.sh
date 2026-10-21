#!/usr/bin/env bash
set -euo pipefail
STORAGE="${STARVLA_STORAGE:-/root/autodl-tmp/starvla}"
TARGET="${STARVLA_RESTORE_TARGET:-/root/workspace/starVLA}"
if [[ -e "$TARGET/reproduction/env.sh" ]]; then
  echo "Preparation workspace is already present: $TARGET/reproduction"
  exit 0
fi
if [[ -e "$TARGET" ]]; then
  echo "Refusing to overwrite an existing path: $TARGET" >&2
  exit 1
fi
(cd "$STORAGE" && sha256sum -c preparation_backup.tar.gz.sha256)
mkdir -p "$TARGET"
tar -xzf "$STORAGE/preparation_backup.tar.gz" -C "$TARGET"
if [[ "$TARGET" == /root/workspace/starVLA ]]; then
  bash "$TARGET/reproduction/setup_local_paths.sh" --legacy-alias
else
  bash "$TARGET/reproduction/setup_local_paths.sh"
fi
echo "Restored $TARGET; run: source $TARGET/reproduction/env.sh"
