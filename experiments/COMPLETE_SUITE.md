# 锁定的CIFAR-10-C完整实验

## 计划与费用

15种官方扰动，Level5，训练种子42、43、44，Original、Original-Reuse、No-history、Ours四模式，共180组完整适应。顺序由complete_protocol.schedule固定：四轮各45组，每轮交错扰动和种子，并轮换模式位置。不是根据准确率调整顺序或参数。

RTX3080实际单价0.93元/小时。按旧16组整进程耗时估计，180组适应、164次新评估（另16组复用旧准确率）需9.1～11.4小时，约8.5～10.6元；准备、上传、归档和等待时间另计，实例退还前仍会计费。开始前~/work可用约25GB，最终权重、日志和归档预计2～4GB。每组只保存最终一份权重，不保存每批中间权重；失败尝试原样保留。低于2GiB可用空间时停止，不自动删除历史文件。

## 算法锁定

不改官方online.py、utils、model、dataset或scripts。online_complete.py来自已通过16组的online_final.py，只有两处执行设置变化：允许指定42/43/44；仅最后一批保存。AST单元测试验证还原这两处后与旧入口一致。batch128、extend8、K50（实际查询52）、memory10000、lr0.001、iters1、repeat1、workers2和原始图、CG、损失、优化器、PMA均固定。

Original保留历史且每批新资源；Original-Reuse只复用资源；No-history只切断历史读取；Ours组合两者。全部模式保持全量入库和每批全新空Flat索引。共享资源不跨进程、线程或实验组。

每个适应过程独立初始化模型、优化器及记忆库。适应种子42/43/44；最终评估始终使用固定种子42+原图索引，维持历史配对评估输入，避免把评估增强噪声加入训练种子差异。eval、inference_mode及参数/BN摘要检查不变，八视图仍用概率平方后跨8×10最大值。

## 旧16组的复用边界

用户明确选择最终一次保存，因此全部180组适应计时必须重新按同一策略测量，旧每批覆盖保存的耗时绝不混算。对已有4扰动seed42的16组，先检查旧文件清单、代码、参数、源权重；新运行结束后，逐张量对比旧模型并核对79批输入/RNG、近邻和伪标签。完全一致才复制旧评估结果和预测，省去16次评估；任何不一致立即失败并留痕，不自动跳过或选用旧准确率。

## 数据及版本

prepare_complete_data.py重新校验官方完整包MD5，仅提取剩余11个普通npy成员，保留全部等级。verify_complete_data.py对新文件逐项SHA256，对旧4文件使用既有校验记录，核验全部15种形状、标签顺序和官方加载器Level5长度。

新代码位于独立分支experiments/locked-15c-3seeds。protocol.json锁定实际全部源码SHA256、15个数据SHA256、源权重SHA256、环境包清单及180个任务ID。Git基线提交和未提交状态单独记录，实际代码以source-snapshot.tar.gz和文件摘要为准；不能只用旧提交号代替新增文件。代码或环境改变则拒绝续跑，不自动推送GitHub。

## 入口与断点

先完成17项单元测试、编译及check_complete_smoke小测。小测包含四模式seed42各8批与旧小测逐项一致，以及Ours种子43/44各一批；每个进程只保存一次。

```bash
bash experiments/start_complete_tmux.sh \
  "$HOME/work/ist-complete-20261117-run02" \
  "$HOME/work/ist-final-stage1-20261117-run01" ist-complete-suite
```

断线不重启。查看tmux或status.json、逐组failure.json及外部-suite.log。确定无进程后，使用同一个ROOT和LEGACY再次执行入口：完整组校验权重/预测后跳过；已适应成功仅评估中断时只续评估；不完整适应保留旧attempt并从预训练起点另建attempt。锁防止同套并发。

任何执行失败都记录任务ID、阶段、时间和traceback并停止。不会删失败或挑选有利结果。首次编译阶段发现的管理脚本字段重名已在正式运行前修复；早期run01小测记录独立保留，最终使用代码锁定后的run02小测。

## 计时与统计

正常适应仅起止CUDA同步，含取数、训练、日志、PMA和最终一次权重保存，排除初始测试及最终配对评估。FAISS为原始整块墙钟，不重复叠加传输；显存为半秒设备采样峰值。完整场景四模式齐全后验证输入/RNG、两对复用的近邻/伪标签/权重以及评估预测CSV完全一致。

每扰动每模式报告三种子均值±样本标准差（ddof1）。总体先求每种子15扰动平均，再对三个种子统计；配对差按同扰动同种子计算，主表再按种子聚合。不得将45个场景冒充45次独立随机种子。

完成180组及45套等价检查后生成PAPER_TABLES.md、paper-main-table.csv、paper-2x2-ablation.csv、statistics.json和逐组per-run.csv。archive_complete自动保存包含权重、日志、源码及SHA256清单的ZIP；任何中断均保留可恢复状态。Gaussian Noise等四种已用于开发，其余11种为新增泛化验证。无三种子统计结果前不宣称最终提升或非劣效。
