# 完整复现步骤与历史边界

所有安装、下载、验证GPU和实验命令都是未来实验机上的操作说明。本轮只在Mac进行CPU语法、源码摘要、命令及指标复核，没有执行下面的GPU/安装步骤。

## 1. 分开发布目录和锁定目录

先按根README创建发布工作树及标签工作树，并设置`RELEASE_DIR`、`LOCKED_DIR`。锁定标签精确提交见`reproduction/manifests/provenance.json`。检出时关闭自动行尾转换，避免改变源码摘要。

不要把release_tools/或原24执行器复制进LOCKED_DIR。旧验证器检查README与上游是否一致，源码清单会记录新增.py/.sh；在发布分支关闭断言不属于原实验。正确做法是只在锁定目录运行原算法，外部工具通过--repo引用它。源码核验为纯CPU操作：

```bash
python3 "$RELEASE_DIR/release_tools/verify_locked.py" --repo "$LOCKED_DIR"
```

## 2. 环境准备（未来Linux实验机）

目标：Linux x86_64，单张RTX3080 10GB，Python3.10.13、torch1.13.1+cu117、torchvision0.14.1+cu117、FAISS GPU1.7.4、NumPy1.23.5、SciPy1.10.1、MKL2021.4匹配构建。CUDA11.4 FAISS库和torch cu117需实机联合验证，不能仅由版本表证明兼容。

以锁定源码里的部署方案为起点；已有conda需由用户提供，脚本不安装conda。先审查再明确安装到不存在的独立前缀：

```bash
export IST_ENV_PREFIX="$HOME/work/envs/ist310"
cd "$LOCKED_DIR"
bash deploy/01_install.sh --dry-run
# 仅在另行决定安装后执行，已有环境不能覆盖：
bash deploy/01_install.sh --apply
```

候选explicit、pip-only及冲突记录见`reproduction/environment/README.md`。正式pip-freeze含构建机file:///路径，不可直接全表安装。没有完整、同一时点且经实机验证的传递依赖锁，不承诺离线重建。

## 3. 官方数据与源权重（自行获取，不进Git）

- 干净CIFAR-10：[官方Python压缩包](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)，170498071字节，MD5 `c58f30108f718f92721af3b95e74349a`。
- CIFAR-10-C：[Zenodo记录2535967](https://zenodo.org/records/2535967)，完整`CIFAR-10-C.tar`，2918471680字节，MD5 `56bf5dcef84df0e2308c6dcbcbbd8499`。获取全部15种普通扰动npy并保留全部五个等级，不裁成只有severity5。
- 源权重：官方AugMix WRN-40-2，脚本`bash "$LOCKED_DIR/deploy/00_fetch_weights.sh"`从固定上游提交获取并校验，SHA-256 `a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82`。这不是随机初始化，也不是新的训练随机种子。

将官方数据放在锁定目录忽略的data/，也可在data/不存在时链接外部数据目录。不要多套一层data/data；不要向发布目录提交这些文件：

```text
data/
  cifar-10-batches-py/
    data_batch_1 ... data_batch_5
    test_batch
    batches.meta
  CIFAR-10-C/
    gaussian_noise.npy ... jpeg_compression.npy（完整15种）
    labels.npy（官方附带；模型加载使用干净CIFAR标签）
  ModelZoo/
    Hendrycks2020AugMixWRN.pt
```

在环境准备完成后执行：

```bash
python3 "$RELEASE_DIR/release_tools/verify_inputs.py" --repo "$LOCKED_DIR"
"$IST_ENV_PREFIX/bin/python" "$LOCKED_DIR/deploy/02_verify.py" --repo "$LOCKED_DIR" env
"$IST_ENV_PREFIX/bin/python" "$LOCKED_DIR/deploy/02_verify.py" --repo "$LOCKED_DIR" weights
"$IST_ENV_PREFIX/bin/python" "$LOCKED_DIR/deploy/02_verify.py" --repo "$LOCKED_DIR" data
```

第一个工具只做CPU摘要/存在性检查，不证明标签关系；原data验证包含Gaussian Noise及官方干净CIFAR完整性，不能冒充全部15类的新数据加载验证。原`verify_complete_data.py`需历史数据校验记录，不作为从零命令。新包装执行前再核对所选扰动和权重摘要，官方加载器仍执行其原有检查。

## 4. 无旧归档也能生成/启动的新复现

根README中的prepare_runs.py默认只生成命令，Original/Ours或四模式均可从源权重独立开始，无需旧16组和历史smoke。只有在完成上述环境和输入验证后，才可使用实验环境解释器、指定同一--python并追加--execute。示例会产生新的GPU实验，**本轮未执行**：

```bash
"$IST_ENV_PREFIX/bin/python" "$RELEASE_DIR/release_tools/prepare_runs.py" \
  --repo "$LOCKED_DIR" --python "$IST_ENV_PREFIX/bin/python" \
  --output "$HOME/work/ist-fresh-2x2-unique" \
  --corruptions gaussian_noise --seeds 42 --execute
```

删去--execute即只生成计划；--modes Original Ours选两组；--all-180选完整15扰动×3适应随机种子×4模式。新包装默认不续跑，拒绝已有结果目录；失败记录保留，不能将不完整权重当新预训练起点。

命令沿用`complete_protocol.command`的完整适应参数及相对调度：batch128、八视图、memory10000视图节点、K50（L2实际检索52）、lr0.001、iters1、repeat1、workers2、原SGD与批末PMA、仅最终保存。默认master port16835，需确保无冲突，不并发启动同一端口。每组重新加载同一源权重，评估始终固定42+原图索引。

新包装使用原monitor/适应入口/独立评估器，但不复刻旧总控的全部跨组等价审计或旧16组准确率复用；每个新模型都重新评估。`replication-results.json`有publication_wrapper标记，plan中记录锁定提交，不能冒用原180组的complete证明。GPU执行路径尚未验收，源码锁一致不保证新硬件/负载/环境中的位级和耗时一致。

## 5. 原180组历史流程（需要旧归档）

以下只适用于已有完整旧16组、旧smoke和全部校验清单/权重者，不能作为全新用户最短入口。固定环境与工作树后，审阅旧脚本再使用原参数：

```bash
cd "$LOCKED_DIR"
# 这些路径必须是真实、完整的历史归档；示例不会替你生成证明。
"$IST_ENV_PREFIX/bin/python" -m experiments.check_complete_smoke \
  --root "$HOME/work/new-complete-smoke" --legacy-smoke "$HOME/work/真实旧smoke目录"
bash experiments/start_complete_tmux.sh \
  "$HOME/work/new-complete" "$HOME/work/真实旧16组目录" ist-complete-suite
```

总控读取ROOT-smoke/complete.json、旧16组config/results/SHA清单等。恢复旧结果还要求当时源码、环境冻结和路径一致；上传了源码不代表这些旧权重也随Git发布。旧16组耗时采用不同保存策略，禁止混入正式180组，只在权重/轨迹一致时复用准确率。

## 6. 原24组效率流程（需要正式模型归档）

`reproduction/original/efficiency24_runner.py`保留当时执行器原字节。它需固定home/work目录、RTX3080、输入恢复包、完整正式检查点和参考审计轨迹；缺失时不能仅用reference-results的指标JSON代替。执行器在锁定工作树之外，运行前需核对导入路径、数据/模型位置与恢复等待条件。本发布不提供绕过权重一致性断言的一键开关，也不把另写的新脚本称为已运行的原执行器。

24组固定适应随机种子42、三次计时重复；准确率复用而非新增评估。Gaussian Noise第2次减速保留。原报告生成器也保留，但它有固定路径且会写CSV，建议用新的analyze_results.py在全新目录重算。

## 7. 评估、计时与统计

- 最终单视图使用预处理原图；固定八视图使用42+索引生成共享视图，概率平方后在8×10中取最大位置并对10取余，不是平均投票。评估无PLCA、eval/inference_mode、参数及BN不更新。
- 适应墙钟仅完整适应起止CUDA同步，含取数、多视图前向、PLCA、梯度更新、PMA、日志及最后一次保存；不含初始测试和最终评估。FAISS整块是内部阶段，不能与适应时间相加，也不能把局部降幅当整体降幅。显存为设备半秒采样峰值。
- 正式表先按种子平均15扰动，再对3种子求均值与样本标准差；配对减少先逐种子计算再汇总。24组独立逐对比值后对12对求均值/样本标准差，不混入180组。
- CPU统计示例见根README，输入为已发布的小型指标副本，输出必须为新目录。原图像、权重和大日志不在Git中；仅指标的CPU统计不能替代模型/输入轨迹的重新验收。
