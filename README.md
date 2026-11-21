# IST课程项目：当前批次图传播与FAISS GPU资源复用

这是基于Jing Ma的[IST官方实现](https://github.com/JingInAI/IST4TTA)开展的课程复现与效率优化项目，**不是原论文官方仓库**。原论文：[Improved Self-Training for Test-Time Adaptation, CVPR 2024, 23701–23710](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Improved_Self-Training_for_Test-Time_Adaptation_CVPR_2024_paper.html)。原作者版权及根目录MIT LICENSE保留。

## 先看版本

- `release/paper-final`：课程说明、发布工具、原执行器副本、小型指标与复现证据所在分支。
- `paper-experiments-180-24-locked`：提交A的锁定实验版本，**114文件、92份.py/.sh源码摘要与正式快照完全一致**。精确提交号及摘要见[provenance.json](reproduction/manifests/provenance.json)。以提交号固定内容，不移动/重写该实验版本；尚未设置GitHub服务端标签保护规则。
- 必须在独立目录检出锁定标签运行实验。新版README不替换锁定版README，新增发布工具不参与历史源码清单。这解决了旧验证器把README纳入官方一致性检查的限制。

## 最终方案与结果

1. **No-history PLCA**：传播节点仅来自当前批次的八视图，不读取历史节点；仍保留全量历史入库。这是结构简化，会改变算法行为。
2. **FAISS GPU资源复用**：进程内跨批保留`StandardGpuResources`，每批仍新建空的精确Flat L2索引；不复用索引内容。在已验证的相同历史配置下保持预测结果。

保留原多视图、PLCA、PMA、CE+KL损失及优化配置。CE使用PLCA校正后硬化的one-hot标签，KL方向为校正前软概率p到当前预测q，即D_KL(p‖q)，两系数均为1。源码入口为`online_complete.py`，四模式映射为`experiments/final_protocol.py`，两项改动见`experiments/history.py`和`faiss_runtime.py`。

| 方法 | 历史读取 | GPU资源复用 | 固定八视图准确率/% | 适应耗时/s |
|---|---|---|---:|---:|
| Original | 是 | 否 | 89.861±0.011 | 180.58±5.61 |
| Original-Reuse | 是 | 是 | 89.861±0.011 | 159.77±4.00 |
| No-history | 否 | 否 | 89.829±0.015 | 141.68±5.86 |
| Ours | 否 | 是 | 89.829±0.015 | 123.01±10.35 |

以上仅为**180组正式实验**：CIFAR-10-C，15扰动、severity=5、4模式、3个适应随机种子42/43/44，单张RTX3080 10GB、WRN-40-2。三个种子从同一AugMix源权重独立开始；先在种子内平均15扰动，再对三个均值计算均值±样本标准差（ddof=1）。Original指锁定入口的原配置，不直接拿每批保存的旧`online.py`计时代替。

Ours相对Original的种子层配对时间减少为**31.92%±4.59%**：R_s=100×(1−T̄_Ours,s/T̄_Original,s)，先计算三个R_s再求均值和样本标准差，不能改为总体均值之比。八视图准确率降低约0.032个百分点，不代表统计等效或非劣效。

**24组独立效率重复**为4扰动×2模式×3次计时重复，固定适应随机种子42，同样使用RTX3080；不是增加3个训练模型种子。全部12对平均配对加速1.67±0.55倍、时间减少33.95%±23.19%；11对更快、1对更慢。高斯噪声第2次151.15→187.83秒，减速24.27%，全部保留。其准确率经权重/轨迹核验后复用正式评估，不是24次重新评测。两套实验不合并；RTX3060仅做后期检查备份。

## 快速开始：只做CPU核验与统计

以下示例在远端已有发布分支和标签后可用，也可用等价的本地仓库路径克隆。CPU核验无需安装环境、下载数据或运行GPU。

```bash
git -c core.autocrlf=false clone --branch release/paper-final https://github.com/Reedy123/IST4TTA-course-project.git ist-course-release
cd ist-course-release
export RELEASE_DIR="$PWD"
git -c core.autocrlf=false worktree add --detach ../ist-experiment-locked paper-experiments-180-24-locked
export LOCKED_DIR="$(cd ../ist-experiment-locked && pwd)"
python3 "$RELEASE_DIR/release_tools/verify_locked.py" --repo "$LOCKED_DIR"
python3 "$RELEASE_DIR/release_tools/analyze_results.py" \
  --locked-repo "$LOCKED_DIR" \
  --formal-results "$RELEASE_DIR/reproduction/reference-results/formal-results.json" \
  --efficiency-results "$RELEASE_DIR/reproduction/reference-results/efficiency24-results.json" \
  --output "$RELEASE_DIR/../ist-cpu-statistics-new"
```

只需现有Python3标准库，不安装环境、不读模型或数据集。统计输出必须是全新目录；保留全部12对和减速样本。Python版本不同可能产生浮点末位误差，不能因此修改原CSV/JSON。

## 从零运行Original、Ours及2×2：先生成计划

准备步骤见[完整复现说明](docs/REPRODUCTION.md)。需自行获取数据和源权重，并在Linux x86_64、单张RTX3080上通过环境验证。普通Mac可以生成下面的计划，**默认不会运行GPU**：

```bash
python3 "$RELEASE_DIR/release_tools/prepare_runs.py" \
  --repo "$LOCKED_DIR" --output "$HOME/work/ist-new-original-ours" \
  --modes Original Ours --corruptions gaussian_noise --seeds 42

# 四模式2×2：不指定--modes时默认四组，先从同一扰动/种子核对。
python3 "$RELEASE_DIR/release_tools/prepare_runs.py" \
  --repo "$LOCKED_DIR" --output "$HOME/work/ist-new-2x2" \
  --corruptions gaussian_noise --seeds 42

# 完整180组新复现计划，采用原调度；打印计划不代表已经运行。
python3 "$RELEASE_DIR/release_tools/prepare_runs.py" \
  --repo "$LOCKED_DIR" --output "$HOME/work/ist-new-180" --all-180
```

实际新运行需在已验证的实验环境中使用相同命令、显式追加`--execute`并指定`--python "$IST_ENV_PREFIX/bin/python"`；启动包装本身也使用该环境解释器。**该发布包装仅通过CPU检查，尚未做GPU执行验收**。它重新评估每个新模型，无旧16组依赖、无自动续跑、不替代原总控完整审计链，也不保证精度/耗时逐位重现。输出`replication-results.json`是新复现数据，不能冒充原180组或与历史24组混合。

## 环境、数据和原历史执行器

- [环境材料及冲突说明](reproduction/environment/README.md)：锁定版使用MKL2021.4匹配构建；候选导出仍有不一致记录，未宣称完整验证环境锁。
- [从零准备、计时/评估及历史续跑边界](docs/REPRODUCTION.md)：官方数据获取地址、摘要、目录及验证命令；不随Git分发数据或模型。
- [原始执行器与来源](reproduction/README.md)：原24执行器和原报告脚本保留不改；旧180总控需要旧16组归档和smoke证明，不能称为一键从零复现。
- [发布工具](release_tools/README.md)：明确默认只读行为、输入/输出限制及新包装的范围。
- [探索代码分类](docs/EXPLORATION.md)：保留探索材料，不把原图级图、筛选/门控、加权或CG调整列入最终贡献。

## 发布范围与限制

保留原项目Git历史，因此历史仍可访问原作者约8.62 MiB预训练权重；当前树及新增提交不包含模型权重。`.gitignore`不能移除历史对象。本轮不删除历史、不强推、不更改远端默认分支。原数据集、模型、实验大包、环境目录及凭据不进入新提交；这里只发布小型指标和协议。

本项目仅覆盖有限数据集、模型、三个适应随机种子及少量计时重复。适应后离线固定模型评估不等同严格在线TTA；资源局部耗时下降不能直接充当端到端下降；不能声称每种扰动/每次运行均更优或稳定加速。课程提交作者身份尚未提供，未在本仓库虚构作者资料。
