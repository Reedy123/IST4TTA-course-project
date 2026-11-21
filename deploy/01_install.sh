#!/usr/bin/env bash
# 在 Linux 独立前缀安装环境；默认仅做 conda 求解预览。
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
PREFIX=${IST_ENV_PREFIX:-"$HOME/work/envs/ist310"}
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo '仅支持 Linux x86_64'; exit 1; }
command -v conda >/dev/null || { echo '缺少 conda，请先确定环境管理方式'; exit 1; }
[[ ! -e "$PREFIX" ]] || { echo "环境路径已存在，拒绝覆盖：$PREFIX"; exit 1; }
MODE=${1:---dry-run}
[[ "$MODE" == --dry-run || "$MODE" == --apply ]] || exit 2
EXTRA=(--dry-run)
[[ "$MODE" == --apply ]] && EXTRA=(-y)
conda create -p "$PREFIX" --override-channels -c pytorch -c nvidia -c defaults   --strict-channel-priority --file "$KIT/conda-spec.txt" "${EXTRA[@]}"
[[ "$MODE" == --apply ]] || { echo '预览完成；确认后使用 --apply'; exit 0; }
conda run -p "$PREFIX" python -m pip install --no-cache-dir -c "$KIT/constraints.txt" -r "$KIT/requirements.txt"
conda run -p "$PREFIX" python -m pip check
mkdir -p "$KIT/deployment-record"
conda list -p "$PREFIX" --explicit > "$KIT/deployment-record/conda-explicit.txt"
conda run -p "$PREFIX" python -m pip freeze > "$KIT/deployment-record/pip-freeze.txt"
# 单独记录 pip 来源包，避免重建时覆盖 conda 的 FAISS 和数值库。
conda list -p "$PREFIX" --json > "$KIT/deployment-record/conda-list.json"
conda run -p "$PREFIX" python -c 'import json,sys; rows=json.load(open(sys.argv[1])); print("\n".join(x["name"]+"=="+x["version"] for x in rows if x.get("channel")=="pypi" or x.get("build_string")=="pypi_0"))' "$KIT/deployment-record/conda-list.json" > "$KIT/deployment-record/pip-only-lock.txt"
echo "安装完成，环境前缀：$PREFIX；请运行环境验证。"
