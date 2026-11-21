"""只读汇总固定图回放，计算标签翻转与已记录的CG时间预算。"""
import csv
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse.linalg._isolve import iterative,utils
from .cg_replay import dump,LIMITS,scores_to_labels


def main(root):
    """不读取模型，不运行新求解或训练。"""
    root=Path(root);graphs=json.loads((root/'all-graphs.json').read_text());summary=[]
    for limit in LIMITS:
        stages=[next(s for s in g['phases'] if s['limit']==limit) for g in graphs];res=np.array([v for s in stages for v in s['relative_residual']])
        item=dict(limit=limit,relative_residual_median=float(np.median(res)),relative_residual_max=float(res.max()),view_correct=sum(s['metrics']['view_correct'] for s in stages),group_correct=sum(s['metrics']['group_correct'] for s in stages),different_labels_vs20=sum(s['different_from_20'] for s in stages),different_labels_vs_direct=sum(s['different_from_direct'] for s in stages),native_median_seconds_sum=sum(s['plain_median_seconds'] for s in stages),converged_systems=sum(v==0 for s in stages for v in s['info']),
                  view_wrong_to_right=sum(s['metrics']['view_wrong_to_right'] for s in stages),view_right_to_wrong=sum(s['metrics']['view_right_to_wrong'] for s in stages),group_wrong_to_right=sum(s['metrics']['group_wrong_to_right'] for s in stages),group_right_to_wrong=sum(s['metrics']['group_right_to_wrong'] for s in stages))
        summary.append(item)
    cases=[];paired=dict(view_disagreements=0,view_wrong_to_right=0,view_right_to_wrong=0,group_disagreements=0,group_wrong_to_right=0,group_right_to_wrong=0)
    for g in graphs:
        step=g['batch']
        with np.load(root/'fixed-systems'/f'{step:03d}.npz') as a,np.load(root/'replay'/f'{step:03d}-labels.npz') as b:
            old=b['reference20'];new=b['direct_labels'];truth=np.repeat(a['truth'],8);p=a['p'];idx=a['original_indices']
            for i in np.flatnonzero(old!=new):cases.append(dict(batch=step,view_node=int(i),original_test_index=int(idx[i//8]),augmentation_index=int(i%8),truth=int(truth[i]),label20=int(old[i]),direct_label=int(new[i])))
            paired['view_disagreements']+=int((old!=new).sum());paired['view_wrong_to_right']+=int(((old!=truth)&(new==truth)).sum());paired['view_right_to_wrong']+=int(((old==truth)&(new!=truth)).sum())
            old_group=(p*np.eye(10)[old]).reshape(-1,80).argmax(1)%10;new_group=(p*np.eye(10)[new]).reshape(-1,80).argmax(1)%10;y=a['truth']
            paired['group_disagreements']+=int((old_group!=new_group).sum());paired['group_wrong_to_right']+=int(((old_group!=y)&(new_group==y)).sum());paired['group_right_to_wrong']+=int(((old_group==y)&(new_group!=y)).sum())
    with (root/'changed-labels20-vs-direct.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['batch','view_node','original_test_index','augmentation_index','truth','label20','direct_label']);writer.writeheader();writer.writerows(cases)
    iterations=[s['iterations'] for g in graphs for s in g['native_200']];plateaus=[g['stability']['offline_plateau_candidate'] for g in graphs]
    prior=Path.home()/'work/ist-edge-reliability-20261117T033732Z';analysis=json.loads((prior/'analysis.json').read_text())['groups']['baseline']
    result=dict(graphs=8,raw_images=1024,views=8192,stages=summary,paired20_vs_direct=paired,
                raw=dict(view_correct=sum(g['raw']['view_correct'] for g in graphs),group_correct=sum(g['raw']['group_correct'] for g in graphs)),
                direct=dict(view_correct=sum(g['direct']['metrics']['view_correct'] for g in graphs),group_correct=sum(g['direct']['metrics']['group_correct'] for g in graphs),maximum_relative_residual=max(g['direct']['maximum_relative_residual'] for g in graphs)),
                actual_numerical_iterations=dict(minimum=min(iterations),median=float(np.median(iterations)),maximum=max(iterations)),
                stability=dict(permanent_label_iterations=[g['stability']['first_permanent_match_to_final'] for g in graphs],plateau_candidates=plateaus,graphs_with_later_flip_after_plateau=sum(g['stability']['any_later_change_after_plateau'] for g in graphs),graphs_with_plateau_labels_different_from_final=sum(g['stability']['plateau_differs_from_final']>0 for g in graphs),graphs_stopped_before20=sum(k is not None and k<20 for k in plateaus),note='未来永久稳定时刻是事后统计，不是可部署判据'),
                matrix=dict(symmetry_relative_frobenius_min=min(g['symmetry_relative_frobenius'] for g in graphs),symmetry_relative_frobenius_max=max(g['symmetry_relative_frobenius'] for g in graphs),minimum_eigenvalue_symmetric_part=min(g['min_eigenvalue_symmetric_part'] for g in graphs)),
                historical_full_budget=dict(cg_seconds=analysis['cg']['cg_seconds'],adaptation_seconds=analysis['adaptation_seconds'],cg_fraction=analysis['cg']['cg_seconds']/analysis['adaptation_seconds'],note='来自先前同口径Baseline日志，不是本轮新适应运行；完全消除CG才达到该预算上限'),
                no_training=True,limits_selected_by_accuracy=False)
    dump(root/'summary.json',result);(root/'scipy-system-source.txt').write_text(inspect.getsource(utils.make_system));(root/'scipy-atol-source.txt').write_text(inspect.getsource(iterative._get_atol));print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main(sys.argv[1])
