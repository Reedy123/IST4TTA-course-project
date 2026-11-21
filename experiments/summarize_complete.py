"""生成完整实验逐组表、三种子统计、论文主表及二乘二消融表。"""
import argparse
import csv
import json
from pathlib import Path
import statistics
from .complete_protocol import CORRUPTIONS, SEEDS
from .final_protocol import MODES

METRICS=['single_percent','eight_percent','adaptation_seconds','faiss_seconds','peak_sampled_gpu_MiB']


def moment(values):
    """样本标准差使用ddof=1；不足三种子不伪造完整统计。"""
    return dict(n=len(values),mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None)


def csv_table(path, rows):
    """按统一键顺序写可直接用于论文排表的CSV。"""
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def fmt(a):
    """表格明确显示三种子均值和样本标准差。"""
    return f"{a['mean']:.2f} ± {a['std']:.2f}"


def summarize(root, rows, complete=False):
    """部分进度只出逐组表；完整180组才生成主结果和消融统计。"""
    flat=[{k:r[k] for k in ['id','corruption','seed','mode',*METRICS,'checkpoint_seconds',
          'legacy_accuracy_reused','legacy_timing_reused','checkpoint_sha256']} for r in rows]
    csv_table(root/'per-run.csv',flat)
    if not complete:return
    assert len(rows)==len({r['id'] for r in rows})==180
    by={(r['corruption'],r['seed'],r['mode']):r for r in rows}
    for c in CORRUPTIONS:
        for s in SEEDS:
            for m in MODES:assert (c,s,m) in by
    statistics_by_corruption=[]
    for c in CORRUPTIONS:
        for m in MODES:
            stats={k:moment([by[c,s,m][k] for s in SEEDS]) for k in METRICS}
            statistics_by_corruption.append(dict(corruption=c,mode=m,metrics=stats))
    aggregate={}
    seed_means={}
    for m in MODES:
        seed_means[m]={s:{k:statistics.mean(by[c,s,m][k] for c in CORRUPTIONS) for k in METRICS} for s in SEEDS}
        aggregate[m]={k:moment([seed_means[m][s][k] for s in SEEDS]) for k in METRICS}
    paired=[]
    for a,b in [('Original','Original-Reuse'),('Original','No-history'),('No-history','Ours'),('Original-Reuse','Ours'),('Original','Ours')]:
        cases=[]
        for c in CORRUPTIONS:
            metrics={k:moment([by[c,s,b][k]-by[c,s,a][k] for s in SEEDS]) for k in METRICS}
            cases.append(dict(corruption=c,metrics=metrics))
        delta={k:moment([seed_means[b][s][k]-seed_means[a][s][k] for s in SEEDS]) for k in METRICS}
        reduction=moment([100*(1-seed_means[b][s]['adaptation_seconds']/seed_means[a][s]['adaptation_seconds']) for s in SEEDS])
        paired.append(dict(contrast=a+' -> '+b,definition='后者减前者；准确率单位百分点，耗时单位秒',
                           seed_aggregate_difference=delta,adaptation_reduction_percent=reduction,by_corruption=cases))
    result=dict(complete=True,groups=180,seeds=SEEDS,corruptions=15,ddof=1,
                aggregation='总表先按每种子平均15扰动，再统计三个种子；不将45个场景当作45个独立随机种子',
                by_corruption=statistics_by_corruption,aggregate=aggregate,paired=paired)
    (root/'statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    main=[]
    for c in CORRUPTIONS:
        row=dict(corruption=c)
        for m in MODES:
            d=next(r for r in statistics_by_corruption if r['corruption']==c and r['mode']==m)['metrics']
            for k in METRICS:
                row[m+' '+k+' mean']=d[k]['mean'];row[m+' '+k+' std']=d[k]['std']
        main.append(row)
    csv_table(root/'paper-main-table.csv',main)
    ablation=[]
    for m,(history,reuse) in MODES.items():
        row=dict(mode=m,history=history=='original',faiss_reuse=reuse=='reuse-resources')
        for k in METRICS:
            row[k+' mean']=aggregate[m][k]['mean'];row[k+' std']=aggregate[m][k]['std']
        ablation.append(row)
    csv_table(root/'paper-2x2-ablation.csv',ablation)
    text=['# CIFAR-10-C Level5主实验','',
          '完整15扰动×3种子×4模式，共180组。所有适应耗时采用仅最终一次保存协议；旧16组只复用经权重逐张量验证的准确率，旧耗时不混入本表。', '',
          '## 主表：固定八视图准确率（%，三种子均值 ± 样本标准差）','',
          '| 扰动 | Original | Original-Reuse | No-history | Ours |','|---|---:|---:|---:|---:|']
    for c in CORRUPTIONS:
        cells=[]
        for m in MODES:
            d=next(r for r in statistics_by_corruption if r['corruption']==c and r['mode']==m)
            cells.append(fmt(d['metrics']['eight_percent']))
        text.append('| '+c+' | '+' | '.join(cells)+' |')
    text+=['| 15扰动平均 | '+' | '.join(fmt(aggregate[m]['eight_percent']) for m in MODES)+' |','',
           '## 二乘二消融：先求每种子15扰动均值，再统计三种子','',
           '| 模式 | 历史读取 | 资源复用 | 单视图% | 八视图% | 适应秒 | FAISS秒 | 采样显存MiB |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for m,(h,r) in MODES.items():
        text.append('| '+m+' | '+('是' if h=='original' else '否')+' | '+('是' if r=='reuse-resources' else '否')+' | '+' | '.join(fmt(aggregate[m][k]) for k in METRICS)+' |')
    text+=['','## 配对差异：后者减前者','',
           '| 对照 | 单视图变化pp | 八视图变化pp | 适应时间变化秒 | 适应时间减少% |',
           '|---|---:|---:|---:|---:|']
    for p in paired:
        d=p['seed_aggregate_difference']
        text.append('| '+p['contrast']+' | '+fmt(d['single_percent'])+' | '+fmt(d['eight_percent'])+' | '+fmt(d['adaptation_seconds'])+' | '+fmt(p['adaptation_reduction_percent'])+' |')
    text+=['','## 协议与限制','',
           '评估增强固定为42+原始索引，训练种子42/43/44。八视图沿用概率平方后跨8×10最大值，无PLCA最终评估；全程eval和inference_mode，权重与BN不变。', '',
           '适应起止CUDA同步，含取数、日志、PMA和最终一次权重保存；不含初始测试及最终评估。FAISS为相同整块墙钟，含资源和索引创建、add/search及传输。显存为半秒采样设备峰值。', '',
           '任务按固定清单分轮交错扰动、种子和模式，所有成功和失败尝试均保留。系统负载和同步盘噪声仍可能影响耗时；仅三种子，不宣称统计非劣效或普遍加速。', '',
           'Ours是既定当前批次PLCA＋FAISS资源复用，不含新加权、门控、K调整或求解器更改。Gaussian Noise等四种扰动已用于开发，其余11种为新增泛化场景，不能将全部15种称为完全未见验证。', '',
           '旧4扰动16组的每批保存计时与本协议不同，原始记录独立保留，不与主表混算。源码、源权重、数据摘要、环境与180组清单在protocol.json及source-snapshot.tar.gz锁定。']
    (root/'PAPER_TABLES.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    rows=json.loads((args.root/'results.json').read_text())
    summarize(args.root,rows,complete=len(rows)==180)
