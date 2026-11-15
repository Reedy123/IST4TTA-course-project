#!/usr/bin/env bash
# 从固定官方提交获取权重，校验后才放入加载器指定路径。
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$KIT/.." && pwd)
TARGET="$REPO/data/ModelZoo/Hendrycks2020AugMixWRN.pt"
EXPECTED=a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82
mkdir -p "$(dirname "$TARGET")"
if [[ -f "$TARGET" ]]; then
  printf '%s  %s\n' "$EXPECTED" "$TARGET" | sha256sum -c -
  exit 0
fi
TMP=$(mktemp "${TARGET}.download.XXXXXX")
trap 'rm -f "$TMP"' EXIT
curl -L --fail --retry 2 --connect-timeout 30 \
  https://raw.githubusercontent.com/JingInAI/IST4TTA/03ba8e716f9bb6e0a586718f8836362b7090692b/data/ModelZoo/Hendrycks2020AugMixWRN.pt -o "$TMP"
printf '%s  %s\n' "$EXPECTED" "$TMP" | sha256sum -c -
mv "$TMP" "$TARGET"
