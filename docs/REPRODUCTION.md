# 从零复现与历史实验边界

新入口是`python reproduce.py`。它不读旧16组、不假定`/home/featurize`或固定环境前缀；使用当前Python，另建锁定源码工作树、链接用户的数据，结果只写全新目录。原204组归档和锁定标签不改变。

## 1. 克隆和环境

需要Git完整克隆、Linux x86_64、已有conda及可用NVIDIA GPU。Mac可进行CPU核验/生成计划，不能执行原CUDA/FAISS GPU模型。ZIP下载或浅克隆可能缺少锁定提交及公开权重的历史对象，因此建议完整克隆：

```bash
git -c core.autocrlf=false clone --branch release/paper-final https://github.com/Reedy123/IST4TTA-course-project.git ist-course
cd ist-course
python3 reproduce.py install --prefix ../ist-env
# 审阅计划后，仅在准备新环境时加--apply。已有实验室环境不重装/覆盖。
python3 reproduce.py install --prefix ../ist-env --apply
conda activate "$(cd ../ist-env && pwd)"
```

版本和修复前/后证据见[环境说明](../reproduction/environment/README.md)。采用Python3.10.13、torch1.13.1+cu117/torchvision0.14.1+cu117、FAISS GPU1.7.4、MKL2021.4、NumPy1.23.5、SciPy1.10.1；安装器使用修复后的明确包URL和pip来源锁。驱动能力CUDA13.0、torch运行库11.7、FAISS conda Toolkit11.4.1是不同字段，不相互覆盖。新机器安装尚未在本轮实测。

## 2. 公开数据与权重

需要官方干净CIFAR-10与CIFAR-10-C完整包。下载后先检查大小/MD5，再解压；保留完整五个severity，不裁npy。以下Linux命令只适用于尚不存在的下载/数据目录，已有合格数据可跳过下载直接setup。

```bash
mkdir ../ist-downloads ../ist-data
curl -L --fail --retry 2 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o ../ist-downloads/cifar-10-python.tar.gz
curl -L --fail --retry 2 https://zenodo.org/api/records/2535967/files/CIFAR-10-C.tar/content -o ../ist-downloads/CIFAR-10-C.tar
printf '%s  %s\n' c58f30108f718f92721af3b95e74349a ../ist-downloads/cifar-10-python.tar.gz | md5sum -c -
printf '%s  %s\n' 56bf5dcef84df0e2308c6dcbcbbd8499 ../ist-downloads/CIFAR-10-C.tar | md5sum -c -
# CIFAR-10为170498071字节，CIFAR-10-C为2918471680字节。
tar -xzf ../ist-downloads/cifar-10-python.tar.gz -C ../ist-data
tar -xf ../ist-downloads/CIFAR-10-C.tar -C ../ist-data
python reproduce.py weights --data-dir ../ist-data
python reproduce.py setup --data-dir ../ist-data
python reproduce.py inputs --corruptions gaussian_noise
```

[干净CIFAR官方来源](https://www.cs.toronto.edu/~kriz/cifar.html)、[CIFAR-10-C Zenodo](https://zenodo.org/records/2535967)。`weights`从克隆历史中固定上游提交03ba8e7提取公开AugMix WRN-40-2，不使用我们适应后的模型；SHA-256为`a39f020c48b3bad58df794b5d3a286724f00b9f026ee22d0e6d1710748599e82`，已有错误文件拒绝覆盖。若本地缺少该Git对象，可从[固定官方文件](https://raw.githubusercontent.com/JingInAI/IST4TTA/03ba8e716f9bb6e0a586718f8836362b7090692b/data/ModelZoo/Hendrycks2020AugMixWRN.pt)获取，保存到数据目录的ModelZoo同名文件，并用inputs核验，不跳过摘要。

最终外部目录应为：

```text
ist-data/
  cifar-10-batches-py/   # data_batch_1..5、test_batch、batches.meta
  CIFAR-10-C/           # 完整15种npy及官方labels.npy
  ModelZoo/Hendrycks2020AugMixWRN.pt
```

inputs核验源权重及所选扰动SHA、干净数据存在性；官方加载器在运行时检查干净CIFAR完整性。完整15种用`python reproduce.py inputs`。旧verify_complete_data需要历史校验文件，不作为从零入口。

setup默认使用仓库旁`../ist-experiment-locked`，固定提交A并验证114/92摘要，再将该目录的data链接到指定外部数据。若自定义`--runtime /新路径`，后续run/inputs需同时指定`--repo /新路径`。已有非锁定目录或不同数据链接会停止，不覆盖任何文件。不要复制发布脚本进锁定工作树。

## 3. 检查设备，先跑小测

先确认已获得实验室空闲GPU的使用授权/调度分配。脚本只在启动前检查GPU进程、显存和利用率，不能阻止其他用户随后启动；没有调度锁或独占保证。忙碌、信息不全、环境不符就停止，不杀进程、不覆盖环境。诊断后无计算进程、显存不超过256 MiB但利用率暂高时，最多复查5秒；连续两次（间隔1秒）达到原空闲阈值才启动。有进程、显存超限或读数无效时立即停止；持续利用率超过5%则超时停止。`gpu-safety.jsonl`记录UTC时间、显存、利用率、PID及复查结果。该检查没有调度独占保证。

```bash
python reproduce.py check
# 此命令进行微型联合GPU算子检查，不是模型测试；先确保卡空闲。
python reproduce.py check --require-gpu --device 0
# 以下是真正GPU新运行；删除--execute则只打印计划。
python reproduce.py run --mode Original --smoke --output ../smoke-original --execute
python reproduce.py run --mode Ours --smoke --output ../smoke-ours --execute
python reproduce.py run --smoke --output ../smoke-2x2 --execute
```

默认小测适应2批（256张原图，batch128、八视图保持不变），初始评估和最终评估各128张。`--max-batches 3 --eval-images 256`可显式调整小测。为避免原入口初始评估全部10000张，外部`adapt_smoke_entry.py`仅在内存中给唯一test_dataset加固定子集；适应流程/损失/更新不改文件。最终小测评估复用锁定的SharedViews、predict、state_hash函数，固定原索引0..N-1；不是论文完整准确率，也不与历史180/24组混合。

其他空闲NVIDIA卡可验证功能，但记录实际硬件，`paper_hardware=false`时不能视为RTX3080论文效率复测。新入口已完成[限定范围四模式GPU小测](GPU_SMOKE_ACCEPTANCE.md)，成功的CPU模拟本身不能称为GPU复现成功。

## 4. 完整适应与2×2

```bash
python reproduce.py run --mode Original --output ../full-original --execute
python reproduce.py run --mode Ours --output ../full-ours --execute
python reproduce.py run --output ../full-2x2 --execute
# 完整180组仅生成计划；本次任务禁止实际重跑。
python reproduce.py run --all-180 --output ../new-180-plan
```

去掉--smoke时，直接调用锁定online_complete.py和complete_evaluate.py，适应/评估均为10000张，无初始评估内存适配。保留batch128、extend8、K50（实际检索52）、memory10000视图、lr0.001、iters1、repeat1、workers2、原CE/KL/SGD/PMA、最后一批保存。Original指该锁定原配置，不是旧每批保存入口的耗时。

可用`--mode Original-Reuse No-history`、`--corruptions fog jpeg_compression`、`--seeds 42 43 44`选择组合；其他参数为--device、--master-port、--repo和--python。适应随机种子均从同一源权重开始，评估增强始终42+原索引。原调度用于过滤后的相对顺序，不使用旧16组准确率；每个新模型独立重新评估。没有自动续跑，失败保留日志，需要新目录开始。

## 5. 输出和统计

```text
新输出根目录/
  plan.json                   # 源码锁、发布工具摘要、任务与实际命令
  inputs.json / environment.json
  gaussian_noise-s42-Original/
    command.json / adaptation.log / gpu.csv / launcher.json
    adaptation/               # profile、审计与唯一最终*adapt.pth
    evaluation-command.json / evaluation.log
    evaluation/results.json / predictions.csv
  replication-results.json
  complete.json               # 全部所选任务检查通过才创建
  failure.json                # 失败时保留，不能当完成证明
```

适应秒数取原profile墙钟，排除初始/最终评估；包含取数、PLCA、模型更新、PMA、日志和最终保存。FAISS秒数是内部阶段，不能再和适应相加。设备显存半秒采样峰值不是瞬时精确峰值。新监控器只采所选物理GPU，子进程映射cuda:0，不误用服务器其他卡。

```bash
python reproduce.py summary --root ../smoke-2x2 --output ../smoke-summary
python reproduce.py summary --root ../full-2x2 --output ../full-summary
```

生成逐组CSV、方法汇总CSV及JSON。按每个适应随机种子先平均所选扰动，再对种子汇总；只有一个种子时样本标准差留空，不伪造重复。小测只报告流程指标，不输出论文加速宣称。新结果带publication_wrapper、smoke_only及实际硬件标记，不能替换论文/PPT或与历史效率记录混算。

## 6. 原流程仍保留，但不是新同学入口

- 原180组总控继续需要旧16组config/results/SHA、旧smoke证明和特定路径；原源码及标签一字不改。
- 原24执行器还需要正式权重及轨迹逐项一致，复制准确率而非重新评估。没有这些归档不能把小型指标JSON当作模型。
- 新入口不关闭旧断言，而在独立锁定工作树之外提供包装。完整运行不用小测AST适配；小测改动明确记录，未声称与原正式执行逐位/耗时一致。
- 旧README验证限制通过独立A工作树解决，发布README可完善但锁定README保持上游原字节。

2026-11-21补充验收：33项CPU测试通过；授权连接恢复后，RTX 3080原环境完成四模式高斯噪声小测（适应256张、初始/最终各评估128张）。诊断后利用率采样尾迹的误拦截已由20db3a8新增工具修复，未改原环境/归档/锁定源码；未重跑正式实验，完整规模和新环境安装仍未验证。详见[验收记录](GPU_SMOKE_ACCEPTANCE.md)。
