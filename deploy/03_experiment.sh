#!/usr/bin/env bash
# 仅供完成三项验证后手动启动正式 Gaussian Noise Level 5 实验。
set -euo pipefail
REPO=$(cd "${1:?请提供官方仓库目录}" && pwd)
OUT=${2:?请提供仓库以外的全新结果目录}
[[ "$OUT" == /* && ! -e "$OUT" ]] || { echo '结果目录必须是不存在的绝对路径'; exit 1; }
[[ "$OUT" != "$REPO" && "$OUT" != "$REPO/"* ]] || { echo '结果必须存放在官方仓库之外'; exit 1; }
# 允许协作提交，要求原始版本是祖先且算法目录保持一致。
git -C "$REPO" merge-base --is-ancestor 03ba8e716f9bb6e0a586718f8836362b7090692b HEAD
git -C "$REPO" diff --exit-code 03ba8e716f9bb6e0a586718f8836362b7090692b -- model dataset utils online.py online_clip.py scripts requirements.txt LICENSE README.md
[[ -z $(git -C "$REPO" status --porcelain) ]] || { echo '官方仓库不干净'; exit 1; }
mkdir -p "$OUT"
cd "$REPO"
export PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch   --nproc_per_node 1 --master_port "${IST_MASTER_PORT:-16341}" online.py   --dataset cifar10-c --dataset_path ./data/ --corruption gaussian_noise --level 5   --model WRN-40-2 --pretrained --num_classes 10 --iters 1 --batch_size 128   --num_workers "${IST_NUM_WORKERS:-2}" --save_model --save_path "$OUT" 2>&1 | tee "$OUT/console.log"
