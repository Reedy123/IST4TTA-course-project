# 历史读取贡献对照

用途：online_history.py支持--history_mode original/no-history，默认original。只允许admission=original全量入库。官方online.py和其他入口不变。

no-history仅在repeat>0适应时禁止读取历史特征和标签，当前批次PLCA、K50、损失、SGD、PMA、八视图以及历史写入不变。repeat=0最终评估不改变。新增pseudo_quality.jsonl只用真值作离线诊断，不参与训练决策。

专项测试验证原版路径保持对象、no-history不调用get_all，以及诊断不修改随机状态。完整唯一新增对照已验证79批10000张、历史读取全0、输入与随机状态与原Original基线一致。固定评估85.34%单视图、87.10%八视图；Original为85.39%、87.12%。历史边际收益很小，不能由单种子证明普遍无效。

详见持久化结果~/work/ist-no-history-20261115T141107Z/。本轮没有其他GPU实验、参数搜索或推送。
