> 协作仓库使用说明：以根目录 DEPLOY.md 为准。工具已整合到 deploy/，原始源码保持不变；下文原工具包目录示例仅供参考。验证脚本允许新增协作提交，并核对原始算法差异。

# IST 最小复现部署材料

用途：在 Featurize Linux x86_64、RTX 3080 10GB 上部署独立 Python 3.10 环境。官方仓库保持原样，材料和结果都放在仓库外。

检查日期：2026-11-15。未安装任何依赖、未执行验证或实验、未连接服务器。当前仅完成源码检查、发布元数据核对和脚本语法检查。版本约束不是经过服务器验证的完整环境锁。

## 检查结论

- 官方 commit：03ba8e716f9bb6e0a586718f8836362b7090692b，工作区干净。
- 必须安装：torch、torchvision、GPU FAISS、NumPy、SciPy、timm、Pillow、PyYAML、tqdm 及其必要传递依赖。
- 本实验可省略：CLIP、ftfy、regex、datasets、torchaudio。源码导入链不依赖它们；不要执行官方 requirements.txt 的整表安装。
- FAISS 包来自 pytorch 渠道；它使用的 CUDA Toolkit 11.4.1 来自 nvidia 渠道。此前只写 pytorch/defaults 不完整。
- 选定 libfaiss 要求 cudatoolkit >=11.4.1,<11.5 和 mkl>=2021。固定 MKL 2023.1.0；其传递依赖含 intel-openmp 2023.* 和 tbb 2021.*。避免求解到 MKL 2024.1+，旧 PyTorch 生态有 iJIT_NotifyEvent 导入错误报告，但不能将该报告直接视为本组合必现。
- NumPy 1.23.5 同时满足 FAISS 的 <2 与 SciPy 1.10.1 的 >=1.19.5,<1.27；SciPy 1.10.1 保留 cg(tol=...)。
- FAISS CUDA 11.4 与 PyTorch wheel cu117 是两套用户态运行库，不需要替换服务器驱动或系统 CUDA。现有 cu117 可用是有利条件，但不证明动态链接顺序安全。验证脚本在两个独立进程中测试两种导入顺序，并检查 GPU 运算、原版 PLCA 半精度稀疏计算及 NCCL。
- 不设置或覆盖 LD_LIBRARY_PATH。若云镜像已设置 CUDA 路径，先记录；动态库失败时可在子进程中用 env -u LD_LIBRARY_PATH 重试并比较，不能直接删除系统库。
- 源码无需修改。保持 batch=128、extend=8、max_len=10000、iters=1、seed=42；num_workers 默认用 2，减少进程资源压力。

## 本地文件状态及下载清单

已存在：官方仓库、WRN 权重（约 8.6MB），摘要见 PROVENANCE.txt。

仍缺失：

1. 原版 CIFAR-10 Python 数据集。官方地址：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 。解压后应为 REPO/data/cifar-10-batches-py/，完整保留 data_batch_1 至 data_batch_5、test_batch、batches.meta 等官方文件；torchvision 的完整性检查会检查多个文件。
2. CIFAR-10-C。官方记录：https://zenodo.org/records/2535967 ，下载 CIFAR-10-C.tar；本次只需从中保留 CIFAR-10-C/gaussian_noise.npy，放到 REPO/data/ 下。必须保留原始 50000 张，不能只存 Level 5。

当前加载器先调用原版 CIFAR-10 test，再将图片替换为 gaussian_noise.npy[40000:50000]；标签沿用原版测试集。labels.npy 未被读取，可保留作数据归档，但本次不是运行必需文件。无需 CIFAR-100、ImageNet、其他损坏类型和额外权重。本次未下载上述数据。

## 下次租机的最短顺序

先上传官方仓库与本材料目录，二者保持独立。建议目录：~/ist-work/IST4TTA-official 和 ~/ist-work/ist-minimal-kit。所有以下命令仅为待执行步骤。

```bash
export REPO="$HOME/ist-work/IST4TTA-official"
export KIT="$HOME/ist-work/ist-minimal-kit"
export IST_ENV_PREFIX="$HOME/ist-work/env-ist310"
export PYTHONDONTWRITEBYTECODE=1
uname -m
ldd --version | head -1
nvidia-smi
conda --version
```

成功标准：x86_64、glibc>=2.17、GPU 为 RTX 3080 且驱动正常、conda 可用。失败则先处理镜像/环境管理工具，勿安装到原 Python 3.8 环境。本材料不自动安装 conda。

### 阶段一：环境安装与验证

```bash
bash "$KIT/01_install.sh" --dry-run
bash "$KIT/01_install.sh" --apply
conda activate "$IST_ENV_PREFIX"
python "$KIT/02_verify.py" env --repo "$REPO"
```

成功标准：求解包含指定 FAISS/libfaiss/CUDA/MKL、pip check 成功、两种导入顺序都通过、合成 PLCA 和 NCCL 通过，最后输出“环境验证通过”。这一步不加载模型训练。

失败处理：

- 包找不到：确认三个渠道可访问，检查代理和 conda 索引；不要改装 CPU FAISS 或 nightly。需重新核对包是否仍可获取。
- 已存在前缀：脚本拒绝覆盖；若这是失败安装，保留记录并换一个新的独立前缀重试。
- iJIT_NotifyEvent：核对 conda list 中的 MKL 是否为 2023.1.0；不要升级 torch 作为临时绕过。
- CUDA/FAISS 动态库错误或仅一种导入顺序通过：记录驱动、conda list、LD_LIBRARY_PATH 和完整报错，停止后续阶段。可用 env -u LD_LIBRARY_PATH 在子进程重试；仍失败则需要重新设计统一 CUDA 构建方案，不能宣称部署成功。
- NCCL 错误：核对是否分配了 GPU、驱动和容器支持，保留错误；不修改算法或直接跳过验证。
- PLCA 稀疏算子失败：保留栈信息，说明库版本表虽然匹配但该运行路径不兼容，停止实验。

### 阶段二：模型权重验证

```bash
python "$KIT/02_verify.py" weights --repo "$REPO"
```

成功标准：权重 SHA256 正确、严格加载成功、GPU 推理输出为 (2,10) 和 (2,128)，数值有限。无反向传播。

失败处理：校验失败则重新从固定 commit 取权重；路径失败则检查仓库上传位置；不使用随机初始化或 strict=False 绕过。

### 阶段三：数据验证

准备上述两个数据集后：

```bash
python "$KIT/02_verify.py" data --repo "$REPO"
```

成功标准：原版数据完整性检查通过、损坏文件为 uint8 (50000,32,32,3)、Level 5 正好 10000 张、标签与原版测试标签一致。

失败处理：重新下载或修正目录；保持完整官方文件，不能用裁剪后的损坏文件。脚本先以 download=False 检查原版 CIFAR-10，缺失时失败，不会替用户下载。

### 阶段四：正式实验（以上全部通过后才执行）

```bash
bash "$KIT/03_experiment.sh" "$REPO" "$HOME/ist-work/results-gaussian5-run1"
```

成功标准：进程退出码 0、约 79 个在线批次处理完毕、日志中有初始和最终 Target Dataset Accuracy、结果目录有适应后 pth、无 NaN/Traceback。最终指标采用官方增强/伪标签评估口径，不是普通单次原图准确率，也不是严格逐样本 prequential 指标。与论文数值是否一致需另行核对实验协议。

失败处理：显存不足先检查其他 GPU 进程，不自动修改 batch/extend/记忆库；更改这些参数会改变实验设置。共享内存/worker 错误可设置 IST_NUM_WORKERS=0 后使用全新结果目录重跑。端口冲突可设置 IST_MASTER_PORT=16342。不要在失败产生的半成品结果目录上覆盖运行。

## 重复部署与记录

安装脚本首次部署固定直接依赖，pip 的传递依赖仍由解析器选择；当前不能称为完全锁定或已验证环境。首次成功后保存 deployment-record/、所有验证终端输出、nvidia-smi 和 commit。

以后在同平台可使用实际成功环境的 conda-explicit.txt 创建新前缀，再按 pip-freeze.txt 还原 pip 包（注意其中包含 conda 已安装的 NumPy/SciPy/FAISS，不能盲目全表 pip install）。最稳妥方式是复用 conda 显式锁，并单独保存实际 pip 来源包的锁与 wheel 缓存；跨平台不能复用 Linux 显式锁。当前 requirements.txt 配合 constraints.txt 是重建候选方案，成功环境的传递依赖锁需首次服务器部署后生成。

## 来源

- https://github.com/facebookresearch/faiss/blob/v1.7.4/INSTALL.md
- https://api.anaconda.org/package/pytorch/faiss-gpu
- https://api.anaconda.org/package/pytorch/libfaiss
- https://api.anaconda.org/package/nvidia/cudatoolkit
- https://api.anaconda.org/package/anaconda/mkl
- https://github.com/pytorch/pytorch/issues/123097
- https://pypi.org/pypi/scipy/1.10.1/json
- https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.sparse.linalg.cg.html
- https://pytorch.org/get-started/previous-versions/

首次成功后，下一次同平台重建可使用下列实际部署记录（须保存原文件和三个阶段的成功记录）：

```bash
conda create -p "$IST_ENV_PREFIX" --file "$KIT/deployment-record/conda-explicit.txt"
conda run -p "$IST_ENV_PREFIX" python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu117 \
  -c "$KIT/constraints.txt" -r "$KIT/deployment-record/pip-only-lock.txt"
```

pip-only-lock.txt 来自 conda 的 pip 来源记录，只包含 pip 包；pip-freeze.txt 用于审计。远端历史包仍可能被移除，长期复现还应缓存包文件。
