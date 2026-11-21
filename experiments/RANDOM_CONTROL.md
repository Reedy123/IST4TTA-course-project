# 等入库率随机入库模式

用途：独立online_random.py入口，读取--admission_schedule指定的Confidence逐批JSONL，使用--admission_random_seed（默认42）的独立PCG64随机源，选择相同数量原图入库。原四模式入口和官方算法文件不变。

每组八视图，随机选择后排序恢复原图顺序。配额/实际batch不匹配或配额越界立即报错；最终评估repeat=0不随机筛选。逐批日志保存配额源SHA256、接受索引、容量、类别分布和年龄。input_trace沿用原诊断实现，完整运行验证与Confidence逐批完全相同，不消耗训练的全局随机状态。

本次只运行Gaussian Noise Level5、训练seed42、独立抽样seed42。配额来自既有Confidence实验，不是通用部署策略，不读取测试正确性来选样本。结果为85.47%单视图、87.28%固定八视图；Confidence为85.32%、87.24%。单次结果不支持置信度筛选优于等配额随机筛选。
