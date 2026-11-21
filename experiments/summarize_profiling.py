"""汇总同步诊断与正常耗时，计算不带门控开销的理论节省。"""
from pathlib import Path
import hashlib
import json
import math
import sys
import zipfile
import torch


def main(root):
    """只读既有记录，生成可备份的汇总，不启动GPU工作。"""
    root=Path(root);rows={};graph_names=['plca_setup','faiss','graph','graph_cpu_bridge','scipy_cg','propagation_other']
    for flow in ['no-history','hard-only']:
        normal=json.loads((root/'normal'/flow/'adaptation/profile-summary.json').read_text())
        diag=json.loads((root/'diagnostic'/flow/'adaptation/profile-summary.json').read_text())
        s=diag['stage_exclusive_seconds'];total=diag['adaptation_wall_seconds']
        phases={'eight_view_forward':s['feature_forward'],'faiss':s.get('faiss',0),'graph_build':s.get('graph',0),'graph_cpu_bridge':s.get('graph_cpu_bridge',0),'scipy_cg':s.get('scipy_cg',0),'propagation_other':s.get('propagation_other',0),'plca_setup':s.get('plca_setup',0),'model_update':s['model_update']}
        phases['other']=total-sum(phases.values())
        graph=sum(s.get(k,0) for k in graph_names)
        overhead={k:v for k,v in s.items() if k not in ['feature_forward','model_update']+graph_names}
        overhead['unattributed']=diag['unattributed_other_seconds']
        timing=json.loads((root/'normal'/flow/'launcher-timing.json').read_text())
        rows[flow]=dict(normal_adaptation_seconds=normal['adaptation_wall_seconds'],diagnostic_adaptation_seconds=total,
                        diagnostic_phases_seconds=phases,other_breakdown_seconds=overhead,graph_pipeline_seconds=graph,
                        graph_fraction_of_diagnostic=graph/total,normal_launcher=timing,normal_checkpoint_block_seconds=normal['checkpoint_block_seconds'])
    flow='no-history';row=rows[flow];graph=row['graph_pipeline_seconds'];total=row['diagnostic_adaptation_seconds']
    stages=json.loads((root/'diagnostic'/flow/'adaptation/profile-stages.json').read_text());perbatch={i:0. for i in range(79)}
    for r in stages:
        if r['name'] in graph_names:perbatch[r['batch']]+=r['exclusive_seconds']
    bounds=[]
    for rate in [.25,.5,.75]:
        uniform=rate*graph;count=math.floor(79*rate);best=sum(sorted(perbatch.values(),reverse=True)[:count])
        bounds.append(dict(skip_rate=rate,uniform_ideal_seconds=uniform,uniform_ideal_fraction=uniform/total,
                           uniform_ideal_speedup=1/(1-uniform/total),floor_batch_count=count,most_expensive_batches_upper_seconds=best,most_expensive_batches_upper_fraction=best/total))
    root_parent=root.parent;refs={'no-history':root_parent/'ist-no-history-20261115T141107Z/adaptation', 'hard-only':root_parent/'ist-selective-targets-20261115T155245Z/hard-only/adaptation'}
    equivalence={}
    for flow,path in refs.items():
        now=torch.load(next((root/'normal'/flow/'adaptation').glob('*adapt.pth')),map_location='cpu')
        old=torch.load(next(path.glob('*adapt.pth')),map_location='cpu')
        equivalence[flow]=dict(keys_equal=set(now)==set(old),all_tensors_equal=all(torch.equal(now[k],old[k]) for k in now),different_tensors=sum(not torch.equal(now[k],old[k]) for k in now))
    observed=json.loads((root/'normal-graph/no-history/adaptation/profile-summary.json').read_text())
    graph_normal=observed['stage_inclusive_seconds']['plca_total'];normal_total=observed['adaptation_wall_seconds']
    normal_records=json.loads((root/'normal-graph/no-history/adaptation/profile-stages.json').read_text())
    normal_batches=[r['inclusive_seconds'] for r in normal_records if r['name']=='plca_total'];assert len(normal_batches)==79
    normal_bounds=[]
    for rate in [.25,.5,.75]:
        cost=rate*graph_normal;count=math.floor(79*rate);best=sum(sorted(normal_batches,reverse=True)[:count])
        normal_bounds.append(dict(skip_rate=rate,uniform_ideal_seconds=cost,uniform_ideal_fraction=cost/normal_total,uniform_ideal_speedup=1/(1-cost/normal_total),floor_batch_count=count,most_expensive_batches_upper_seconds=best,most_expensive_batches_upper_fraction=best/normal_total))
    normal_observation=dict(adaptation_seconds=normal_total,graph_seconds=graph_normal,graph_fraction=graph_normal/normal_total,equivalence=json.loads((root/'normal-graph/no-history/equivalence.json').read_text()),note='PLCA whole block wall time, no additional CUDA sync, includes original final labels.cpu() completion; excludes resource teardown after function return')
    results=dict(normal_graph_observation=normal_observation,theoretical_bounds_normal=normal_bounds,rows=rows,theoretical_bounds_diagnostic_only=bounds,diagnostic_and_normal_weights=json.loads((root/'checkpoint-equivalence.json').read_text()),historical_checkpoint_equivalence=equivalence,
                 limitations=['one measurement per mode and timing method; no variance estimate','normal and diagnostic are separate full runs; diagnostic share is not a precise normal-wall share','savings assume zero decision overhead and unchanged update/IO cost; no achieved speedup claimed','first-batch initialization included; no accuracy-based threshold or parameter change'])
    (root/'results.json').write_text(json.dumps(results,indent=2))
    manifest=[dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ['summary-mac.zip','sha256-manifest.json']]
    (root/'sha256-manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(root/'summary-mac.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.pth' and p.name!='summary-mac.zip':z.write(p,str(p.relative_to(root)))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main(sys.argv[1])
