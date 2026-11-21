# 记忆库入库筛选：第一阶段

用途：课程方案 A，独立入口 online_filtered.py；online.py、utils/、模型和损失保持作者原版。分支 feature/memory-admission。未推送。

四模式：original、confidence、consistency、combined；参数 --admission、--confidence_threshold（默认0.8）、--consistency_threshold（默认0.75）。

每张原图的八视图先计算预测众数（平票取较小类别）；置信度为八视图对众数类别的概率均值，一致性为众数票数/8。联合模式要求两阈值同时满足。未读取真实标签作筛选。只筛选适应阶段 memory_bank.update 输入；被接受的八条特征和原有PLCA标签保持对象、顺序和数值。当前批次训练集合、损失、优化器不变。repeat=0的官方最终评估保留原版全量写入，不筛选。

PLCA的soft_labeling按原始视图索引写回，所以reshape为原图×视图×类别；不按随机推理返回顺序分组。original直接将原列表交给原MemoryBank.update，不调用随机函数。

FIFO容量继续采用作者视图级容量，容量不能整除8时可切断最旧组；这与入库前整组选取是两件事。若改为整组淘汰将改变原版语义，本阶段不做。年龄单位是适应批次，从特征提取/入库批次计数，不是秒。空筛选不补样本、不放宽阈值，旧条目正常变老。

日志 admission.jsonl：每批及累计原图入库率、视图级记忆库大小、PLCA标签类别分布、平均及最大年龄、置信度及一致性均值。类别分布不是测试真值分布。

测试：tests/test_admission.py 包含8项测试；tests/smoke_admission.py是明确的两批正确性检查，不用于正式准确率报告。GPU检查为batch32、两批共64张、views8、FIFO1003、workers0，和原版入口临时截断版本对照。original最终所有state_dict张量精确相同，去除进度条重复输出后16次loss记录相同。不能据两批检查推断整程位级一致；正式参数仍应使用batch128、max_len10000。

实际入库：original 64/64=100%，confidence 41/64=64.0625%，consistency 56/64=87.5%，combined 41/64=64.0625%。记忆库分别512、328、448、328视图条目。没有运行完整消融。

建议首组正式参数：阈值0.8及0.75、seed42、batch128、extend8、max_len10000、K50、原SGD及学习率0.001、iters1。统一使用已固定的配对评估器；不要自动根据标签寻优。联合条件当前未显示额外筛选，下一阶段应记录其跨批次交集和拒绝原因再判断。

小规模检查命令：

```bash
PYTHONDONTWRITEBYTECODE=1 ~/work/envs/ist310/bin/python -m unittest discover -s tests -p test_admission.py -v
PYTHONDONTWRITEBYTECODE=1 ~/work/envs/ist310/bin/python tests/smoke_admission.py "$PWD" "$HOME/work/新的smoke结果目录"
```

正式实验必须另行授权。官方deploy/03_experiment.sh仍指向online.py，不会自动使用筛选版。筛选版需明确调用online_filtered.py及其参数，避免污染原始基线。
