# 统一环境版本与证据

本次找到了原MKL修复前/后记录，不再把两份不同阶段的导出当成互相矛盾的正式环境。

| 项目 | 采用版本 | 依据 |
|---|---|---|
| Python | 3.10.13 | 修复后conda清单与锁定conda-spec |
| torch / torchvision | 1.13.1+cu117 / 0.14.1+cu117 | 正式pip-freeze、修复后清单、GPU验证日志 |
| torch CUDA运行库 | 11.7 | 正式verify-env日志；官方cu117 wheel |
| FAISS GPU / libfaiss | 1.7.4，cuda11.4构建 | 修复后conda清单、联合GPU检查 |
| conda cudatoolkit | 11.4.1 | 修复后清单；这是FAISS依赖，不是torch.version.cuda |
| MKL / intel-openmp | 2021.4.0，固定构建 | 修复后清单、锁定源码中环境修复方案 |
| NumPy / SciPy / timm | 1.23.5 / 1.10.1 / 0.6.13 | 正式记录及联合验证日志 |
| 原设备/驱动 | RTX3080 10GB，580.95.05 | 正式gpu-initial；其中CUDA13.0是驱动能力显示，不是torch实际运行库 |

`historical-inconsistent-conda-list.json`的关键版本与新找到的`before-packages.json`相同（MKL2023.1及对应构建），属于修复前记录；`verified-repair-conda-list.json`是修复后完整包列表，MKL为2021.4，和`linux-64.explicit.txt`一致。比较结果、来源SHA和证据范围见`version-evidence.json`；联合GPU成功记录见`verified-repair-gpu-check.log`。不根据文件名猜测阶段。

旧candidate文件为兼容既有发布保留，不作为安装入口；`formal-pip-freeze.txt`原样保留作证据，其中file:///构建路径不可直接pip install。新安装入口统一使用修复后`linux-64.explicit.txt`和按修复后包清单提取的19个pip来源包`pip-runtime.txt`，防止pip覆盖conda管理的FAISS/MKL。安装后导出当前实际环境，不能把旧证据冒充新的安装验证。

```bash
# 只显示命令，无conda也不会安装；prefix必须不存在。
python3 reproduce.py install --prefix ../ist-env
# 仅未来Linux x86_64上确认安装时才执行，不覆盖原实验室环境。
python3 reproduce.py install --prefix ../ist-env --apply
conda activate "$(cd ../ist-env && pwd)"
python reproduce.py check
# 确认设备空闲后，可检查微型GPU算子；不是模型冒烟。
python reproduce.py check --require-gpu --device 0
```

安装器要求已有conda，不安装conda、不改系统CUDA/驱动、不自动删除失败前缀；日志另存`ist-env-install-record`。如果实验室已有符合版本的可用环境，直接激活并检查，跳过安装。

原始修复后联合检查是历史已通过的证据；本次未重装环境、未重新执行GPU算子。服务器当前拒绝连接，新机器的安装、下载包可用性、GPU模型流程仍待验收。explicit仅适用Linux x86_64，不能在Mac/ARM安装。FAISS双库联合运算通过后才有实机兼容证据。

公开安装依据：[PyTorch历史版本（1.13.1 cu117）](https://pytorch.org/get-started/previous-versions/)、[FAISS v1.7.4安装说明](https://github.com/facebookresearch/faiss/blob/v1.7.4/INSTALL.md)。包版本取自本项目历史证据，不升级到当前最新版。
