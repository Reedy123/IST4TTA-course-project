# Featurize 最小复现

本协作仓库基于 JingInAI/IST4TTA，保留作者四条提交、原始 README.md 和 MIT LICENSE。
官方来源：https://github.com/JingInAI/IST4TTA
原始 commit：03ba8e716f9bb6e0a586718f8836362b7090692b。
作者算法文件没有修改，所有部署工具放在 deploy/。新增部署脚本与文档沿用根目录 MIT LICENSE；原作者版权及许可声明完整保留。依赖包各自适用其上游许可证，本仓库不重新分发数据集或依赖软件包。

## 历史与大文件说明

协作分支当前树不跟踪数据或权重。作者历史中已有 8.6MB 权重，为保留原始 commit 未重写历史，因此首次推送仍包含这个历史对象。若要求远端历史也完全没有权重，必须另选不保留原始 commit 的发布方式，不能两者兼得。本次没有推送。

## 服务器步骤

以下命令仅供未来 Linux x86_64 服务器执行。本次未安装、未上机验证。

```bash
# 克隆协作仓库后，进入其根目录
export REPO="$PWD"
export KIT="$REPO/deploy"
export IST_ENV_PREFIX="$(dirname "$REPO")/env-ist310"
export PYTHONDONTWRITEBYTECODE=1

# 将 Featurize 数据集 ZIP 解压到 $REPO/data/，避免嵌套 data/data
# 应存在 data/cifar-10-batches-py/test_batch 和 data/CIFAR-10-C/gaussian_noise.npy
bash "$KIT/00_fetch_weights.sh"
bash "$KIT/01_install.sh" --dry-run
bash "$KIT/01_install.sh" --apply
conda activate "$IST_ENV_PREFIX"
python "$KIT/02_verify.py" env --repo "$REPO"
python "$KIT/02_verify.py" weights --repo "$REPO"
python "$KIT/02_verify.py" data --repo "$REPO"
# 三阶段全部通过后，用仓库外的全新绝对路径启动正式实验
bash "$KIT/03_experiment.sh" "$REPO" "$(dirname "$REPO")/results-gaussian5-run1"
```

conda 必须预先可用。环境方案依然是候选方案，CUDA 11.4/cu117 共存需通过环境验证，失败时停止，不修改算法绕过。完整成功标准、失败处理、依赖说明见 deploy/README.md。新仓库增加协作提交后，验证器检查原始 commit 祖先关系与算法文件差异，不再要求 HEAD 等于作者 commit。

## 提交与推送

协作远程 origin：https://github.com/Reedy123/IST4TTA-course-project 。upstream 保留作者仓库；工作分支为 featurize-minimal。不要向 upstream 推送，也不要使用强制推送。

GitHub CLI 当前未安装，不能确认 CLI 登录状态；浏览器登录和 SSH 鉴权也未验证。不读取或输出凭据。创建远端前复核暂存差异，确认后再配置 origin 和推送。数据、环境、验证记录、结果及凭据不提交。
