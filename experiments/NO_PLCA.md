# no-PLCA软目标机制对照

用途：online_no_plca.py新增--plca_mode original/no-plca。配合--history_mode no-history，从同一AugMix权重独立启动，保持官方online.py和既有入口不变。

no-plca调用官方robust_PLCA的repeat=0路径，保留原视图softmax概率，不再图传播和硬化。目标与soft_labels均为视图数×类别数，原交叉熵和KL公式保持不变；不再重新argmax生成标签。全部原始软目标严格检查，数据、样本、随机数、优化器和PMA与no-history一致。

因此结果反映“图传播及输出硬化”整体作用，不是图拓扑单一变量。no-PLCA固定评估84.19%单视图、86.32%八视图；no-history为85.34%、87.10%。采样峰值显存1723MiB；精确整程墙钟计时因最终汇总错误未写盘，监控跨度约166.42秒，仅供近似参考，不能宣称加速。

3项专项单元测试和256图小规模GPU检查通过；唯一一次10000图完整运行成功，后续CPU汇总已恢复，未重跑GPU。完整记录~/work/ist-no-plca-20261115T145841Z/。
