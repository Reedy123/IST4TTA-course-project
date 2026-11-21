# 最终主线与探索代码分类

本页只提供导航，不移动、删除或修改锁定源码。

| 类别 | 文件 | 使用范围 |
|---|---|---|
| 最终方法/正式运行 | online_complete.py；history.py；faiss_runtime.py；complete_protocol.py；run_complete_suite.py；complete_evaluate.py；summarize_complete.py | 180组和24组共同锁定的主线；实际导入还依赖admission/input_trace/profiling等模块，不能只拷贝两个优化文件 |
| 最终方案早期验证 | online_history.py、online_faiss.py、online_final.py；NO_HISTORY.md、FAISS_REUSE.md、FINAL_SUITE.md | 四扰动/单种子及资源验证历史；计时与评估应按各阶段协议区分 |
| 未纳入最终方案的探索/诊断 | online_filtered.py、online_random.py、online_targets.py、online_no_plca.py、online_weighted_plca.py；cg_replay.py、image_graph_replay.py、edge_reliability.py、target_policies.py等 | 入库筛选、随机对照、选择性纠错、无PLCA、边加权、原图级图及CG重放。保留用于追溯，不作为最终优化贡献 |
| 探索历史文档 | experiments/README.md、CG_REPLAY.md、EDGE_RELIABILITY.md、NO_PLCA.md、RANDOM_CONTROL.md、SELECTIVE_TARGETS.md | 保留当时结论/术语；局部失败或早期准确率不能混入论文正式结果 |
| 新发布工具 | release_tools/与reproduction/ | 未参与历史模型训练；与锁定版本隔离 |

实验源码清单记录的是当时仓库全部.py/.sh，不等于所有文件都构成最终算法。`utils/`、模型与数据加载器保留原实现，包括PMA首次缓存共享张量的原实现行为；本发布没有借整理机会修正算法或扩大结论。
