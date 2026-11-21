# FAISS GPU 资源复用对照

## 范围

独立分支 `performance/faiss-resources`。不改官方 `online.py`、`utils/`、模型、数据集或图传播。独立入口 `online_faiss.py` 默认 `--faiss_mode original`，可切换为 `reuse-resources`。

只把适应调用中的 `StandardGpuResources` 保存在当前 Python 进程、当前调用线程中。索引每批仍使用官方 `GpuIndexFlatL2` 新建，保持默认 float32、K=50（官方 L2 实际查询 K+2=52）、顺序、距离、特征和 PLCA 规则。资源只保存流、cuBLAS 句柄及临时工作区；不保存历史批次的索引数据。

## 为什么只复用资源

原版每批新建资源和索引。资源 Python 构造很便宜，但设备初始化延迟发生在第一次索引创建中。诊断中将 `initializeForDevice(0)` 显式拆出后，已经初始化设备资源的索引创建仅约每批 0.04 ms；因此本轮不增加索引复用/重置的复杂性。

每批新索引必须 `ntotal==0`，add 后必须 `ntotal==当前批视图数`。完整对照记录了全部 79 批计数，包含最后一个 16 原图/128 视图的小批次。跨线程使用缓存会主动拒绝，禁止并发共享同一资源。

## 官方依据

- [FAISS GPU 文档](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)：支持资源对象复用；CPU NumPy 输入由 add/search 自动搬运；资源带有 CUDA 流、句柄和临时工作区。
- [v1.7.4 GpuResources.h](https://github.com/facebookresearch/faiss/blob/v1.7.4/faiss/gpu/GpuResources.h)：未显式初始化时按首次需求分配。
- [v1.7.4 GpuIndexFlat.cu](https://github.com/facebookresearch/faiss/blob/v1.7.4/faiss/gpu/GpuIndexFlat.cu)：Flat 无训练步骤，reset 会清空计数并释放底层存储。本轮不缓存索引，因此无需调用 reset。

这是使用 FAISS 原有资源生命周期接口的工程优化，不是新的近邻搜索算法，不应写成算法创新。

## 文件与流程

- `faiss_runtime.py`：透明调用包装、分项计时、可选资源缓存、每批索引隔离断言与结果哈希。
- `online_faiss.py`：基于既有 No-history 剖析入口的独立运行入口。
- `run_faiss_diagnostic.py`：原版前 8 个真实批次诊断与特征捕获。
- `faiss_device_diagnostic.py`：显式拆分资源初始化；CUPTI 捕获原始调用的传输。
- `check_faiss_reuse.py`：同特征微基准、大小批次切换、两组前 8 批适应等价检查。
- `run_faiss_control.py`：小测通过后仅运行一次完整同期对照及固定配对评估。
- `tests/test_faiss_runtime.py`：生命周期、空索引/脏索引、AST 恢复和训练主体不变检查。

所有函数和类使用中文说明；新增文件不替代既有工作入口。

## 计时口径

小规模诊断在 GPU 阶段前后全设备同步。资源 Python 构造、索引创建（包含设备延迟初始化）、add（含 H2D）、search（含 H2D/D2H）分列。显式 initializeForDevice 仅用于拆分归因，不用于优化正式运行。CUPTI 传输设备时间是 add/search 的子集，不可再叠加。

完整对照用正常计时：只在完整适应起止增加 CUDA 同步，FAISS 整块包围墙钟。search 仍返回官方 NumPy CPU 结果，沿用其自然完成边界。两组都保留每批检查点保存、日志、输入/RNG 哈希和结果审核。整进程时间不用于适应加速结论。显存是 nvidia-smi 每 0.5 秒采样的设备峰值，不是精确瞬时算子峰值。

独立预训练起点、10,000 原图、seed=42、batch=128、extend=8、K=50、memory=10000、lr=0.001、iters=1、workers=2。两组正式训练均跳过原入口最后的额外随机视图评估，改用既有固定配对评估器；不覆盖历史 87.02% 或其他历史结果。

微基准通过标准预先固定为 FAISS 加速至少 1.2 倍且每次节省至少 0.02 秒，然后必须通过近邻/距离/伪标签/权重严格等价。未根据准确率调整参数。只执行一次完整同期对照，整体耗时可能含取数、文件系统与设备调度波动，不能把所有时间差全部归因于资源缓存。

## 复现入口

在项目根目录，以现有 Python 环境运行独立入口；指定全新保存路径：

```bash
~/work/envs/ist310/bin/python -m torch.distributed.launch \
  --nproc_per_node 1 --master_port 16576 online_faiss.py \
  --profile_kind normal --profile_flow no-history \
  --faiss_mode reuse-resources --audit_faiss --skip_final_eval \
  --history_mode no-history --admission original \
  --dataset cifar10-c --dataset_path ./data/ \
  --corruption gaussian_noise --level 5 --model WRN-40-2 \
  --pretrained --num_classes 10 --seed 42 --iters 1 \
  --batch_size 128 --extend 8 --max_len 10000 --lr 0.001 \
  --num_workers 2 --save_model --save_path "$HOME/work/全新结果目录/adaptation"
```

原版仅将 `--faiss_mode` 改成 `original`，使用另一个全新保存目录。正式配置和准确评估命令以当次结果目录里的 `command.json`、`evaluation-command.json` 为准；本轮不重复启动这些命令。
