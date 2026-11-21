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

## 从克隆到运行：新同学流程

**验证状态：原204组实验已完成；新增入口通过33项CPU测试及RTX 3080四模式小规模GPU验收。** 范围为gaussian_noise、severity5、适应随机种子42，每模式适应256张、初始/最终各评估128张。Original沿用4febf16成功记录，其余三组使用20db3a8修复工具；完整10000张流程、新环境安装及其他扰动未在本轮验证。见[验收与安全检查修复记录](docs/GPU_SMOKE_ACCEPTANCE.md)。

### 1. 克隆与准备独立环境

```bash
git -c core.autocrlf=false clone --branch release/paper-final https://github.com/Reedy123/IST4TTA-course-project.git ist-course
cd ist-course
# 先只打印安装计划；需要已有conda。新环境只能用不存在的前缀。
python3 reproduce.py install --prefix ../ist-env
# 在Linux x86_64上决定安装后执行；已有可用环境则跳过安装，直接激活它。
python3 reproduce.py install --prefix ../ist-env --apply
conda activate "$(cd ../ist-env && pwd)"
```

正式记录：Python3.10.13、torch1.13.1+cu117、torchvision0.14.1+cu117、FAISS GPU1.7.4、MKL2021.4.0、NumPy1.23.5、SciPy1.10.1。环境冲突已查明：旧MKL2023清单对应修复前版本；采用修复后明确包清单，见[版本证据](reproduction/environment/README.md)。新机器安装仍须验收，不在Mac安装Linux GPU环境。

### 2. 准备公开数据与权重

完整官方数据下载、MD5核验、解压命令见[数据准备](docs/REPRODUCTION.md#2-公开数据与权重)。已有数据可直接使用仓库外目录，要求含`cifar-10-batches-py/`和`CIFAR-10-C/`，扰动npy保留全部五个等级。

```bash
# ../ist-data是自己准备好的公开数据目录。
python reproduce.py weights --data-dir ../ist-data
python reproduce.py setup --data-dir ../ist-data
python reproduce.py inputs --corruptions gaussian_noise
python reproduce.py check
```

weights优先从完整Git历史中的固定官方提交提取公开AugMix WRN-40-2权重并核验SHA，不读取任何实验适应权重。setup自动在仓库旁创建独立锁定工作树，并链接外部数据；已有源码/数据不会被覆盖。ZIP下载或浅克隆缺历史对象时请按文档完整克隆，或从固定官方URL获取权重。

### 3. 最简Original、Ours与2×2命令

先删除下面命令的`--execute`可只查看计划。实际执行需要已验证的Linux/NVIDIA环境；使用当前Python，无原服务器绝对路径，无旧16组依赖：

```bash
python reproduce.py run --mode Original --smoke --output ../smoke-original --execute
python reproduce.py run --mode Ours --smoke --output ../smoke-ours --execute
# 默认四模式；一次检查Original、Original-Reuse、No-history和Ours。
python reproduce.py run --smoke --output ../smoke-2x2 --execute
```

小测默认适应2批（256张原图），初始/最终各评估128张，仍使用batch128和八视图。小测初始评估通过外部适配器在内存中限制；**不修改锁定源码**，所得准确率/时间只能判断流程有效，不能用作论文结果或稳定加速结论。

### 4. 完整运行与结果

```bash
# 去掉--smoke即完整10000张适应及评估，可只选一种模式。
python reproduce.py run --mode Original --output ../full-original --execute
python reproduce.py run --mode Ours --output ../full-ours --execute
# 同一扰动/种子的完整四模式消融。
python reproduce.py run --output ../full-2x2 --execute
# 仅生成180组计划；不会执行180组。
python reproduce.py run --all-180 --output ../new-180-plan
# 对新运行做CPU汇总，输出必须是另一个全新目录。
python reproduce.py summary --root ../smoke-2x2 --output ../smoke-2x2-summary
```

其他参数：`--corruptions fog jpeg_compression`、`--seeds 42 43 44`、`--device 1`、`--master-port 23123`、`--repo /自己的锁定工作树`、`--python /已有环境/bin/python`。执行前检查所选GPU进程/显存/利用率；忙碌或状态不明就停止，不结束其他任务。其他空闲NVIDIA卡可做功能验证，但会记录非论文3080硬件，不能据此更新论文效率结论；请先通过实验室调度器获得独占或明确授权的空闲卡。

`--output`下保存`plan.json`、环境/输入摘要、每组适应日志、GPU采样、profile、最终权重、评估结果/预测CSV，以及`replication-results.json`、`complete.json`或`failure.json`。结果均为独立新复现。已有输出拒绝覆盖，失败不自动清理/续跑。`summary.csv`给出实际准确率、适应/FAISS秒数，单种子标准差留空。

适应计时与最终评估分开；单视图和固定八视图仍沿用原定义。FAISS是适应内部阶段，不与适应秒数重复相加。完整运行直接执行锁定原入口；小测范围限制不进入正式统计。

## 已验证与未验证的边界

- 历史180/24组与源文件摘要已有归档；原24执行器及旧16组依赖流程保留在[历史说明](reproduction/README.md)，不是新同学从零入口。
- 新入口不读取旧16组、旧服务器目录或旧模型；新每组都重新评估。CPU测试包含真实Git工作树/公开权重SHA、命令/AST、防覆盖和模拟调度；模拟调度不是GPU验证。
- 新入口已在原环境完成上述四模式GPU小测；完整规模及新环境安装仍待实机验收。环境核心版本由日志核验，但公共包可用性、驱动/硬件兼容仍须本机检查，不能宣称已验证所有新机器。
- [探索分类](docs/EXPLORATION.md)保留历史方案，最终贡献不扩大。论文/PPT及原始实验结果不修改。

## 只用CPU核验历史统计

```bash
python reproduce.py stats --locked-repo ../ist-experiment-locked \
  --formal-results reproduction/reference-results/formal-results.json \
  --efficiency-results reproduction/reference-results/efficiency24-results.json \
  --output ../original-statistics-new
python3 -m unittest discover -s release_tools/tests -v
```

只需标准库，无需模型、数据集、GPU或安装新环境。历史指标副本不含原模型参数；统计口径不改变。

## 发布范围与限制

保留原项目Git历史，因此历史仍可访问原作者约8.62 MiB预训练权重；当前树及新增提交不包含模型权重。`.gitignore`不能移除历史对象。本轮不删除历史、不强推、不更改远端默认分支。原数据集、模型、实验大包、环境目录及凭据不进入新提交；这里只发布小型指标和协议。

本项目仅覆盖有限数据集、模型、三个适应随机种子及少量计时重复。适应后离线固定模型评估不等同严格在线TTA；资源局部耗时下降不能直接充当端到端下降；不能声称每种扰动/每次运行均更优或稳定加速。课程提交作者身份尚未提供，未在本仓库虚构作者资料。
