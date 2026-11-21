# 选择性PLCA与硬化对照

用途：online_targets.py独立入口，在feature/selective-plca分支固定no-history，支持--target_mode original/hard-only/selective-plca。官方online.py和先前入口不变。

Hard-only：PLCA关闭，每个原始视图argmax硬化为one-hot。Selective：保留PLCA，当视图原始预测与图提议冲突，且原始最大概率<0.8、同图6/8原始视图支持提议类，才接受图修正；否则保留原始视图argmax。所有训练标签均float32 one-hot，KL原始软目标、损失公式、SGD、PMA、数据及最终评估不变。

可靠性输入来自PLCA前[N,10]概率，按已恢复的视图索引分成原图×8×类别；策略没有真值接口。12项单元测试和两模式各256图GPU检查通过。两组10000图完整运行均匹配基线的输入与随机状态。

固定评估：Hard-only84.31%单视图、86.33%八视图；Selective84.46%、86.45%；No-history85.34%、87.10%。硬化收益相对No-PLCA仅+0.12/+0.01pp，图传播收益相对Hard-only为+1.03/+0.77pp。Selective提出5746次视图修正、接受1105次，未改善完整PLCA。不继续调参或自动扩展跨扰动。

结果保存在~/work/ist-selective-targets-20261115T155245Z/，完整报告随实验和Mac输出归档。开发文件尚未提交，需结合代码快照复现。
