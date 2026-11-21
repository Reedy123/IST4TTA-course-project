# PLCA 邻居软加权原型

## 锁定的定义

独立分支 `feature/plca-edge-reliability`，直接基线为 No-history＋FAISS资源复用。原有官方文件和此前可用入口保持原样。

对传播前每个视图概率 p：

- 置信度 c = 1 − H(p)/ln(C)。
- 留一视图参考 q = 同一原图其余7个视图的概率均值。
- 一致性 a = 1 − JS(p,q)/ln(2)。
- Baseline：h=1；Edge-C：r=c；Edge-V：r=a；Edge-CV：r=c×a。
- 固定 β=0.5，h = 1−β(1−r)，范围[0.5,1]，不搜索参数。
- 原距离权重三次方后，非自环 A_ij 乘邻居端 h_j；真实自环根据邻居索引等于查询索引识别，保持原值，不能假设首邻居就是自身。
- 后续 A+Aᵀ、度归一化、传播右端类别列归一化、CG(tol=1e-6,maxiter=20)、one-hot、训练CE+KL、SGD和PMA均不改。
- 权重计算使用float64概率避免熵消减误差，乘法在float32进行后转回原relation的float16；Baseline和β=0不执行乘法，直接返回原张量。
- 不调用真实标签，不更新模型以生成可靠性，不增加前向或FAISS搜索。历史记忆读取仍关闭；所有组继续原有全量入库。

这属于传播前的边重加权，不是传播后接受或拒绝纠错，不是记忆库筛选。

## 文件

- `edge_reliability.py`：纯概率公式、八视图分组、分布统计及控制组。
- `weighted_plca.py`：在官方源代码的内存副本中插入唯一边权变换及只读记录；复用已验证FaissRuntime。
- `../online_weighted_plca.py`：独立入口，训练循环与online_faiss.py完全一致。
- `../tests/test_edge_reliability.py`：11项新测试，加上既有7项，总计18项。
- `run_edge_smoke.py`：Baseline前8批对历史检查点严格回归，其他三组各检查2批。
- `run_edge_ablation.py`：四组完整10000图像和统一配对评估；仅按预先锁定条件执行控制组。

## 正确性

移除新插入节点及已有透明计时/FAISS委托后，AST必须与官方源文件完全相同。小测Baseline的近邻、距离、伪标签、随机状态及全部模型参数/缓冲区必须与已验证的8批参考完全一致。完整Baseline同样必须与历史完整检查点完全一致。

不同加权组训练后模型本来就会不同，所以不能要求其后续特征、近邻或权重与Baseline一致；所有模式必须保持原始输入、增强张量和全局随机状态一致。

## 日志和计时

- `edge-reliability.jsonl`：每批c/a/r/h分位数、相关性、CV相对C的权重差、实际float16边权改变数量/总量、舍入为零数量、自环不变断言和预测类别分布。
- `reliability-arrays/*.npz`：每个节点的c/a/r/h，供离线汇总。
- `normalized-graph.jsonl`：只读记录实际归一化稀疏图条目、权重分布、自环权重和哈希；不把coalesce副本写回传播矩阵。
- `cg-convergence.jsonl`：原CG返回info、相对残差、有限性及调用耗时；不添加callback或修改参数。
- 原有`pseudo_quality.jsonl`记录伪标签纠错效果，真实标签仅在该离线统计及最终评估中使用，绝不传给权重模块。

原版CG最多20次迭代。新记录发现Baseline也可能全部返回info=20、未达到容差；不能把有限解误写为已收敛，也不为本轮实验增加迭代数。

各组使用正常完整适应计时，起止同步GPU，包含相同观察、统计、存档与每批模型保存。各组均计算c/a分布；因此时间代表带诊断的原型，不与以前无同等日志的运行跨次推断加速。显存来自每0.5秒nvidia-smi采样。

## 实验和控制条件

Gaussian Noise Level 5，10000张；seed42，batch128，extend8，K50（实际查询52），memory10000，lr0.001，iters1，workers2。每组独立初始化模型、优化器、记忆库；固定配对单视图/八视图评估，不运行其他扰动和种子。

仅当Edge-CV满足以下全部条件时，继续Uniform和Shuffled-CV：八视图较Baseline至少增加20个正确预测（0.20个百分点）、严格高于Edge-C和Edge-V、单视图不低于Baseline。该门槛在任何完整结果产生前锁定，只用于是否追加用户授权的控制实验，不是统计显著性或参数寻优。

Uniform使用该组当前预测计算出的CV权重均值；Shuffled-CV按原图八视图块用独立`default_rng(420517)`打乱权重。两者不改变训练全局随机状态。不同组模型轨迹不同，控制组不借用另一组未来的概率或权重。

本轮完整命令、Git基础版本、未提交源码快照、配置、日志、权重均保存到新的`~/work/ist-edge-reliability-时间戳/`。不推送GitHub。
