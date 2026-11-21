"""只汇总新复现入口的结果，小测不计算论文加速结论，不混合历史180/24组。"""
import argparse
import csv
import json
from pathlib import Path
import statistics as st
from runtime_support import check_metrics


def summarize(root, output):
    """要求任务完整且全部有效；按适应随机种子汇总，只有多个种子才计算样本标准差。"""
    root=Path(root).expanduser().resolve();output=Path(output).expanduser().resolve()
    plan=json.loads((root/'plan.json').read_text());complete=json.loads((root/'complete.json').read_text())
    rows=json.loads((root/'replication-results.json').read_text())
    expected={x['id'] for x in plan['tasks']}
    if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected or complete['runs']!=len(rows):
        raise ValueError('不是完整的新复现结果；拒绝把部分运行当成完整结果')
    if not all(r.get('publication_wrapper') and r['smoke_only']==plan['smoke_only'] for r in rows):
        raise ValueError('禁止混入历史正式/效率数据或混合小测与完整运行')
    for row in rows:check_metrics(row['evaluation'],plan['evaluation_images'])
    if output.exists() or output==root or root in output.parents or output in root.parents:
        raise ValueError('汇总输出必须是输入结果目录之外的全新目录')
    metrics=['single_percent','eight_percent','adaptation_seconds','faiss_seconds']
    summary=[]
    for mode in dict.fromkeys(t['mode'] for t in plan['tasks']):
        selected=[r for r in rows if r['mode']==mode];seeds=sorted({r['seed'] for r in selected})
        record={'mode':mode,'runs':len(selected),'adaptation_seeds':len(seeds),'smoke_only':plan['smoke_only']}
        for metric in metrics:
            values=[st.mean(r[metric] for r in selected if r['seed']==s) for s in seeds]
            record[metric+'_mean']=st.mean(values)
            record[metric+'_sample_sd']=st.stdev(values) if len(values)>1 else None
        summary.append(record)
    output.mkdir(parents=True,exist_ok=False)
    fields=['id','mode','corruption','seed','adaptation_images','evaluation_images','single_percent','eight_percent','adaptation_seconds','faiss_seconds','smoke_only']
    with (output/'per-run.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    with (output/'summary.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    report=dict(kind=plan['kind'],runs=len(rows),smoke_only=plan['smoke_only'],summary=summary,
                statistics='每个模式先按适应随机种子平均已选扰动，再对种子取均值和样本标准差；单种子标准差留空',
                timing_note='适应区间与最终评估分开；FAISS为适应内子阶段',
                interpretation='小测仅检查流程，禁止用于论文准确率/加速结论' if plan['smoke_only'] else '独立新复现，不能替换或混合历史论文结果',
                paper_hardware=all(r['hardware']['paper_hardware'] for r in rows))
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    """读取新运行目录并在独立位置生成CSV/JSON，不改原日志和结果。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:print(json.dumps(summarize(args.root,args.output),ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError) as error:parser.exit(2,f'汇总失败：{error}\n')


if __name__=='__main__':
    main()
