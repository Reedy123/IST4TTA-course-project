#!/usr/bin/env bash
# 在tmux中执行或续跑唯一的180组锁定计划，结束后生成论文表格和归档清单。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/work/envs/ist310/bin/python"
ROOT="${1:?需要指定~/work下的结果根目录}"
LEGACY="${2:?需要指定已完成16组结果目录}"
SESSION="${3:-ist-complete-suite}"
case "$ROOT" in "$HOME"/work/*) ;; *) echo '结果路径必须位于~/work'; exit 1;; esac
if tmux has-session -t "$SESSION" 2>/dev/null; then echo 'tmux仍在运行，拒绝重复启动'; exit 1; fi
test -f "${ROOT}-smoke/complete.json"
tmux new-session -d -s "$SESSION" -c "$REPO" bash -c '
set -euo pipefail
repo="$1"; python="$2"; root="$3"; legacy="$4"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
"$python" -u -m experiments.run_complete_suite --root "$root" --legacy "$legacy" >> "${root}-suite.log" 2>&1
"$python" -m experiments.archive_complete "$root" >> "${root}-archive.log" 2>&1
' bash "$REPO" "$PYTHON" "$ROOT" "$LEGACY"
echo "已启动tmux：$SESSION；日志：${ROOT}-suite.log"
