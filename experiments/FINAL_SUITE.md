# 正式第一阶段：当前批次PLCA＋FAISS资源复用

## 四模式

| 模式 | 适应时历史读取 | FAISS资源 |
|---|---|---|
| Original | 作者原始记忆库 | 每批新建 |
| Original-Reuse | 作者原始记忆库 | 进程内复用 |
| No-history | 只用当前批次 | 每批新建 |
| Ours | 只用当前批次 | 进程内复用 |

所有模式全量入库，仅No-history/Ours不读取历史。资源复用仅缓存StandardGpuResources，索引每批新建且必须为空，精确Flat L2和K=50保持不变。Ours不包括此前已否决的原图级图、加权、阈值、选择性纠错或求解器修改。

统一入口 online_final.py 复制已验证online_faiss.py的计算流程，仅增加固定模式映射。官方online.py、utils、model、dataset、scripts保持原样。新增入口的训练循环与原资源复用入口AST逐项一致，官方PLCA内存副本可还原为原始AST。

## 固定实验

Gaussian Noise是开发场景，Motion Blur、Fog、JPEG Compression是初步泛化验证。每种扰动Level5完整10000张、seed42、batch128、extend8、K50（原始实现实际查询52）、memory10000、lr0.001、iters1、repeat1、workers2。每组独立加载相同AugMix WRN-40-2权重和全新优化器、记忆库。PMA和原始CE＋KL损失不变。

统一final_evaluate.py由历史paired_eval.py独立扩展扰动参数；原图单视图及固定8视图共用seed=42+原始索引。八视图仍为概率平方后跨8×10最大值，不改为平均；eval和inference_mode禁止更新参数及BN。评估不执行PLCA，不与旧87.02%的随机评估混为一个口径。

## 数据

Mac执行prepare_final_data.py，先复核完整包官方MD5，再只提取三个普通成员，保留全部等级；逐文件SHA256与包成员一致。服务器verify_final_data.py先校验全部传输SHA256、形状及干净CIFAR标签顺序，再硬链接到现有~/data/CIFAR-10-C，不覆盖已有文件。官方加载器检查四种Level5均10000张。

## 正确性与计时

先运行11项单元测试，再四模式各8批小测。Original/Original-Reuse和No-history/Ours两对近邻索引、距离、输入、增强、RNG、伪标签和权重必须一致。正式16组每一对也重复核验，跨四模式要求输入、增强和RNG轨迹一致。Gaussian正式权重另与历史Original和No-history结果逐张量比较。

适应正常计时只在完整适应起止同步GPU，包含数据等待、日志、输入/RNG/近邻摘要、PMA与每批权重保存；初始测试及最终配对评估均不计入适应。FAISS计时围住相同原始整块，CPU结果返回，含资源创建、索引、add/search及传输，不再叠加子时间。显存为适应进程nvidia-smi半秒采样峰值，不能当作精确瞬时峰值。

不重复正式种子。模式顺序在不同扰动间轮换，减少固定先后偏差，但单次耗时仍受IO/设备调度影响。报告整体、资源优化、去历史三个效应，不能把FAISS局部加速直接当整体加速。

## 启动和续跑

项目根目录执行：

```bash
bash experiments/start_final_tmux.sh "$HOME/work/ist-final-stage1-唯一时间戳" ist-final-stage1
```

查看日志或会话：

```bash
tmux attach -t ist-final-stage1
tail -n 5 "$HOME/work/ist-final-stage1-唯一时间戳-suite.log"
```

断线不用重启。原tmux已结束且确认无残留进程后，使用完全相同结果根路径再次执行启动命令即可。锁禁止同套并发；完成组校验权重后跳过；已完整适应且评估中断时仅续评估；中断适应保留旧attempt，另建新attempt重跑该组，不从不完整权重冒充独立预训练起点。

结果根目录包含配置、Git提交和状态、实际源码SHA256、源码快照、环境版本、逐组命令、完整日志、权重、评估状态及预测CSV、正常计时、显存采样、汇总results.json和summary.csv。all-complete.json仅在全部16组及等价检查通过后生成。完成后停止，不扩展到15扰动，不自动推送GitHub。
