"""只读汇总已完成实验的分布、边变化、纠错及收敛情况，不再运行训练。"""
from pathlib import Path
import json
import sys
import numpy as np
from .edge_reliability import statistics


def records(path):
    """读取每批JSON记录。"""
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main(root):
    """基于全量逐节点记录汇总，保留原始结果不覆盖。"""
    root=Path(root);done=json.loads((root/'complete.json').read_text());assert done['all_complete']
    results=json.loads((root/'results.json').read_text());summary={}
    for mode in done['modes']:
        out=root/'full'/mode/'adaptation';edge=records(out/'edge-reliability.jsonl');cg=records(out/'cg-convergence.jsonl');graphs=records(out/'normalized-graph.jsonl')
        arrays={k:[] for k in ['confidence','agreement','reliability','donor_weight']}
        for file in sorted((out/'reliability-arrays').glob('*.npz')):
            with np.load(file) as r:
                for k in arrays:arrays[k].append(r[k])
        distributions={k:statistics(np.concatenate(v)) for k,v in arrays.items()};assert distributions['confidence']['count']==80000
        ecount=sum(r['nonself_edges'] for r in edge);changed=sum(r['changed_nonself_edges'] for r in edge);before=sum(r['original_nonself_weight'] for r in edge);after=sum(r['actual_nonself_weight'] for r in edge)
        p=results[mode]['profile'];r=results[mode]
        summary[mode]=dict(single_accuracy=r['evaluation']['accuracy']['ist']['single'],eight_accuracy=r['evaluation']['accuracy']['ist']['eight'],adaptation_seconds=p['adaptation_wall_seconds'],faiss_seconds=p['stage_inclusive_seconds']['faiss'],plca_seconds=p['stage_inclusive_seconds']['plca_total'],peak_sampled_gpu_MiB=r['training']['peak_sampled_gpu_MiB'],
                           distributions=distributions,edge=dict(total_nonself_edges=ecount,changed_edges=changed,changed_fraction=changed/ecount,original_weight_sum=before,actual_weight_sum=after,weight_mass_reduction_fraction=1-after/before,observer_calculation_readback_seconds=sum(x['calculation_and_readback_seconds'] for x in edge),positive_edges_rounded_to_zero=sum(x['positive_edges_rounded_to_zero'] for x in edge),self_unchanged=all(x['self_unchanged'] for x in edge),mean_cv_c_weight_difference=sum(x['cv_vs_confidence_weight_mean_difference']*x['views'] for x in edge)/80000),
                           quality=r['quality'],cg=dict(calls=len(cg),info_counts={str(k):sum(x['info']==k for x in cg) for k in sorted(set(x['info'] for x in cg))},relative_residual=statistics([x['relative_residual'] for x in cg]),cg_seconds=sum(x['seconds'] for x in cg)),
                           graph=dict(stored_entries=sum(x['stored_entries'] for x in graphs),nonzero=sum(x['nonzero'] for x in graphs),normalized_self_weight_sum=sum(x['self_weight'] for x in graphs),normalized_weight_sum=sum(x['total_weight'] for x in graphs)))
    trajectory={}
    for first,second in [('baseline','edge-c'),('baseline','edge-v'),('baseline','edge-cv'),('edge-c','edge-cv'),('edge-v','edge-cv'),('edge-c','edge-v')]:
        a=records(root/'full'/first/'adaptation/pseudo-audit.jsonl');b=records(root/'full'/second/'adaptation/pseudo-audit.jsonl')
        trajectory[first+'_vs_'+second]=dict(identical_target_batches=sum(x['pseudo_sha256']==y['pseudo_sha256'] for x,y in zip(a,b)),identical_raw_probability_batches=sum(x['soft_sha256']==y['soft_sha256'] for x,y in zip(a,b)))
    from .run_edge_ablation import paired
    import torch
    extra_paired=paired(root/'full/edge-c/evaluation/predictions.csv',root/'full/edge-v/evaluation/predictions.csv')
    baseline=torch.load(next((root/'full/baseline/adaptation').glob('*adapt.pth')),map_location='cpu');state_differences={}
    for mode in done['modes']:
        state=torch.load(next((root/'full'/mode/'adaptation').glob('*adapt.pth')),map_location='cpu')
        state_differences[mode]=dict(changed_state_tensors=sum(not torch.equal(baseline[k],state[k]) for k in baseline),total_state_tensors=len(baseline))
    output=dict(state_differences_vs_baseline=state_differences,edge_c_vs_v_paired=extra_paired,groups=summary,trajectory=trajectory,controls=json.loads((root/'control-decision.json').read_text()),paired=json.loads((root/'paired-comparisons.json').read_text()))
    (root/'analysis.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main(sys.argv[1])
