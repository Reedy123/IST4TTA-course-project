"""单元测试通过后检查四种模式，Baseline与既有8批检查点严格回归。"""
from pathlib import Path
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import torch


def main():
    """创建全新持久化目录，不覆盖已有结果。"""
    home=Path.home();repo=home/'work/IST4TTA-course-project';root=home/'work'/('ist-edge-reliability-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir()
    tests=home/'work/edge-reliability-unit-20261117.log';assert 'OK' in tests.read_text();(root/'unit-tests.log').write_bytes(tests.read_bytes())
    spec=importlib.util.spec_from_file_location('monitor',home/'work/run_admission_ablation.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    prior=home/'work/ist-faiss-reuse-20261117T013154Z';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0')
    (root/'continuation-rule.json').write_text(json.dumps(dict(beta=.5,seed=42,corruption='gaussian_noise',level=5,eight_gain_over_baseline_minimum_pp=.20,eight_strictly_above_both_single_terms=True,single_not_below_baseline=True,control_modes=['uniform','shuffled-cv'],threshold_search=False),indent=2))
    print('RUN_ROOT='+str(root),flush=True);outcomes={}
    for mode in ['baseline','edge-c','edge-v','edge-cv']:
        group=root/'smoke'/mode;group.mkdir(parents=True);out=group/'adaptation';count=8 if mode=='baseline' else 2
        cmd=json.loads((prior/'full/reuse-resources/command.json').read_text());cmd[cmd.index(str(repo/'online_faiss.py'))]=str(repo/'online_weighted_plca.py');cmd[cmd.index('--max_batches')+1]=str(count);cmd[cmd.index('--save_path')+1]=str(out);cmd+=['--edge_mode',mode,'--capture_faiss']
        (group/'command.json').write_text(json.dumps(cmd,indent=2));print('START smoke '+mode,flush=True)
        timing=monitor.monitor_run(cmd,group/'launcher.log',group/'gpu.csv',repo,env);(group/'launcher-timing.json').write_text(json.dumps(timing,indent=2));assert timing['exit_code']==0
        expected=(prior/'full/reuse-resources/adaptation/input_trace.jsonl').read_bytes().splitlines(keepends=True)
        assert (out/'input_trace.jsonl').read_bytes()==b''.join(expected[:count])
        rows=monitor.read_records(out/'edge-reliability.jsonl');cg=monitor.read_records(out/'cg-convergence.jsonl');assert len(rows)==count and len(cg)==count*10
        assert all(r['solution_finite'] and r['info']>=0 for r in cg)
        assert all(r['self_unchanged'] for r in rows)
        if mode=='baseline':
            assert all(r['changed_nonself_edges']==0 for r in rows)
            reference=prior/'smoke/reuse-resources/adaptation'
            for name in ['input_trace.jsonl','faiss-audit.jsonl','pseudo-audit.jsonl']:
                assert (out/name).read_bytes()==(reference/name).read_bytes(),name
            a=torch.load(next(out.glob('*adapt.pth')),map_location='cpu');b=torch.load(next(reference.glob('*adapt.pth')),map_location='cpu')
            assert all(torch.equal(a[k],b[k]) for k in a),'关闭加权未恢复既有基线权重'
        outcomes[mode]=dict(batches=count,images=count*128,input_rng_exact=True,self_unchanged=True,cg_calls=len(cg),cg_nonconverged=sum(r['info']!=0 for r in cg),baseline_exact=mode=='baseline')
        (root/'smoke-results.json').write_text(json.dumps(outcomes,indent=2));print('DONE smoke '+mode,flush=True)
    (root/'smoke-pass.json').write_text(json.dumps(dict(all_four_passed=True,baseline_checkpoint_exact=True),indent=2));print('SMOKE_COMPLETE '+str(root),flush=True)

if __name__=='__main__':main()
