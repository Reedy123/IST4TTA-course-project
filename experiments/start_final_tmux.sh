#!/usr/bin/env bash
# 在独立tmux会话执行固定小测和正式实验；已完成阶段可安全续跑。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$HOME/work/envs/ist310/bin/python"
ROOT="${1:?请提供全新的持久化结果目录，或原结果目录用于续跑}"
SESSION="${2:-ist-final-stage1}"
case "$ROOT" in "$HOME"/work/*) ;; *) echo '结果必须保存在~/work下'; exit 1;; esac
test -x "$PYTHON"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo '同名tmux会话仍存在，请先查看运行状态'; exit 1
fi
# 通过位置参数传路径，避免将路径当成待执行代码。
tmux new-session -d -s "$SESSION" -c "$REPO" bash -c '
set -euo pipefail
repo="$1"; python="$2"; root="$3"
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1
"$python" -m unittest tests.test_final_protocol tests.test_faiss_runtime tests.test_history -v > "${root}-unit.log" 2>&1
"$python" -u -m experiments.run_final_suite --root "${root}-smoke" --smoke >> "${root}-smoke.log" 2>&1
"$python" -u -m experiments.run_final_suite --root "$root" --validated-smoke "${root}-smoke" >> "${root}-suite.log" 2>&1
' bash "$REPO" "$PYTHON" "$ROOT"
echo "已启动tmux：$SESSION"
echo "日志：${ROOT}-smoke.log 和 ${ROOT}-suite.log"
