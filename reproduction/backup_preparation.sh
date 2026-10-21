#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
REPO="$(dirname "$ROOT")"
STORAGE="${STARVLA_STORAGE:-/root/autodl-tmp/starvla}"
ARCHIVE="$STORAGE/preparation_backup.tar.gz"
# source/ now points to the parent repository, so archive that repository too.
# Symlinks to environments/data are kept as links, never dereferenced.
tar --exclude='./.git' --exclude='./exports' --exclude='./playground' \
  --exclude='*/__pycache__' --exclude='*.egg-info' \
  -czf "$ARCHIVE.tmp" -C "$REPO" .
mv "$ARCHIVE.tmp" "$ARCHIVE"
(cd "$STORAGE" && sha256sum preparation_backup.tar.gz > preparation_backup.tar.gz.sha256)
cp "$ROOT/restore_preparation.sh" "$STORAGE/restore_preparation.sh"
echo "Saved StarVLA code, preparation scripts, configs, and reports to $ARCHIVE"
